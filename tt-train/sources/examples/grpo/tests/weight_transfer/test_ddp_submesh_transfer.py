# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DDP-TTML -> per-submesh-TTT weight transfer test.

Demonstrates the grpo speedup topology where the **TTML side runs DDP**
across N/2 devices (one mesh, weights replicated) and the **TTT side uses
N/2 devices, each in its own [1, 1] submesh** (one model per submesh, the
layout tt-transformers wants for independent data-parallel rollouts).

Launched under tt-run with world_size == 2 (see ``runner_ddp_submesh.sh``):

* **Rank 0 (TTML)** opens a ``[1, 4]`` DDP mesh, loads the instruct
  ``LlamaCompositeKV``, calls ``export_to_hf_dict()`` (replicated weights,
  exactly the contract the transport wants), and ships every tensor to the
  four TTT submeshes over four point-to-point sockets
  (``ttml_i -> submesh_i``). It does **not** need a submesh per device --
  DDP is a single logical mesh.

* **Rank 1 (TTT)** opens a ``[1, 4]`` parent mesh, splits it into four
  ``[1, 1]`` submeshes, builds one :class:`TttGenerationWorker` per
  submesh (dummy boot weights), receives the weights onto each submesh,
  **verifies exact tensor equality** (``torch.equal``) against the
  sender's reference for every transferred tensor, applies them with
  ``update_weights``, and then **prints** (does not assert) each submesh's
  greedy completion for the same prompt so the output can be inspected by
  hand.

The transport itself lives in :mod:`utils.submesh_weight_transfer`.

Self-skips when not launched under tt-run (``OMPI_COMM_WORLD_SIZE`` unset
or != 2). Requires ``HF_TOKEN`` set in the environment (the instruct repo
is gated).
"""

from __future__ import annotations

import gc
import os
import sys
from typing import Any, List

import pytest

_WORLD_SIZE = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "0"))
if _WORLD_SIZE != 2:
    pytest.skip(
        "test_ddp_submesh_transfer must run under tt-run with world_size == 2 "
        "(use tests/weight_transfer/runner_ddp_submesh.sh).",
        allow_module_level=True,
    )

_MPI_RANK = int(os.environ["OMPI_COMM_WORLD_RANK"])

# Fabric is pinned to FABRIC_2D by the autouse session fixture in
# tests/conftest.py (_set_fabric_2d), before any device opens -- the ttml
# rank's enable_fabric() picks FABRIC_2D and the ttt rank must match or the
# cross-rank fabric init collective deadlocks.
from utils.submesh_weight_transfer import (  # noqa: E402
    TTML_RANK,
    TTT_RANK,
    barrier,
    open_receiver_sockets,
    open_sender_sockets,
    recv_weights,
    send_weights,
    verify_received,
)

import ttnn  # noqa: E402

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 32
TEMPERATURE = 0.0  # greedy -> deterministic across submeshes

# 4-devices-per-side config (local8 topology): TTML opens [1, 4] DDP; TTT opens
# [1, 4] and splits into 4 x [1, 1]. For a 2-device debug run, switch to
# grpo_boolq_llama_2dev_ddp_gas_4.yaml + (1, 2) + NUM_SUBMESHES=2 and a matching
# 2-device rank-binding config.
NUM_SUBMESHES = 4
TTML_DEVICE_CONFIG_REL = "tt-train/configs/training_configs/grpo_boolq_llama_4dev_ddp.yaml"
TTT_PARENT_MESH_SHAPE = (1, 4)

# Kept small: the test only generates one prompt per submesh, and four
# worker models share the 4 chips of the TTT rank.
TTT_MAX_BATCH_SIZE = 8
TTT_MAX_SEQ_LEN = 512


# ---------------------------------------------------------------------------
# TTML model loader (mirrors LlamaGRPOCompleter.__init__ -- the HF-download
# branch, which loads replicated weights with no shard mapper).
# ---------------------------------------------------------------------------


def _build_ttml_model(mesh_device: Any, raw: dict, model_source: str, *, enable_ddp: bool):
    import ttml
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from ttml.common.config import get_model_config
    from ttml.models import RunnerType, WeightTyingType
    from ttml.models.llama import LlamaConfig, LlamaRopeScalingConfig, load_from_safetensors
    from utils.llama_grpo_completer import _ensure_safetensors_dir
    from utils.llama_overrides import LlamaCompositeKV

    autograd_ctx = ttml.autograd.AutoContext.get_instance()

    tf_config = get_model_config(raw["training_config"]["model_config"])
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    tf_config.vocab_size = len(tokenizer)

    rope_scaling = LlamaRopeScalingConfig(
        scaling_factor=getattr(tf_config, "scaling_factor", 0.0) or 0.0,
        high_freq_factor=getattr(tf_config, "high_freq_factor", 4.0) or 4.0,
        low_freq_factor=getattr(tf_config, "low_freq_factor", 1.0) or 1.0,
        original_context_length=getattr(tf_config, "original_context_length", 0) or 0,
    )
    runner_type = RunnerType.from_string(str(tf_config.runner_type))
    weight_tying = WeightTyingType.Disabled
    if tf_config.weight_tying:
        weight_tying = WeightTyingType.from_string(str(tf_config.weight_tying))

    llama_cfg = LlamaConfig(
        hidden_size=tf_config.embedding_dim,
        intermediate_size=tf_config.intermediate_dim,
        num_hidden_layers=tf_config.num_blocks,
        num_attention_heads=tf_config.num_heads,
        num_key_value_heads=tf_config.num_groups,
        vocab_size=len(tokenizer),
        max_position_embeddings=tf_config.max_sequence_length,
        rope_theta=tf_config.theta or 10000.0,
        attention_dropout=tf_config.dropout_prob,
        mlp_dropout=tf_config.dropout_prob,
        runner_type=runner_type,
        weight_tying=weight_tying,
        rope_scaling=rope_scaling,
    )

    tt_model = LlamaCompositeKV(llama_cfg)

    # DDP: a single logical mesh with replicated weights + gradient
    # all-reduce. No per-device submesh -- that is the whole point.
    if enable_ddp:
        autograd_ctx.initialize_parallelism_context(ttml.autograd.DistributedConfig(enable_ddp=True, enable_tp=False))

    # HF-download branch: load_from_safetensors uploads with no shard
    # mapper, so every parameter is replicated across the [1, 4] mesh --
    # exactly what export_to_hf_dict + the transport require.
    model_repo_path = snapshot_download(
        repo_id=model_source,
        allow_patterns=["*.safetensors", "*.bin", "*.json", "*.model", "*.txt"],
    )
    model_repo_path = _ensure_safetensors_dir(model_repo_path)
    load_from_safetensors(tt_model, model_repo_path, llama_cfg)
    return tt_model


# ---------------------------------------------------------------------------
# TTML rank
# ---------------------------------------------------------------------------


def _ttml_side() -> None:
    import ttml
    from _completer_utils import close_device, load_device_config, open_device

    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    autograd_ctx.initialize_distributed_context(*sys.argv)

    device_config, raw = load_device_config(TTML_DEVICE_CONFIG_REL)
    mesh_device = open_device(device_config)
    print(
        f"[TTML rank {TTML_RANK}] opened DDP mesh shape={[int(d) for d in mesh_device.shape]} "
        f"enable_ddp={device_config.enable_ddp}",
        flush=True,
    )

    model = None
    sockets = None
    hf_dict = None
    try:
        model = _build_ttml_model(mesh_device, raw, MODEL_ID, enable_ddp=device_config.enable_ddp)

        hf_dict = model.export_to_hf_dict()
        print(f"[TTML rank {TTML_RANK}] export_to_hf_dict -> {len(hf_dict)} tensors", flush=True)

        sockets = open_sender_sockets(mesh_device, NUM_SUBMESHES)
        print(f"[TTML rank {TTML_RANK}] opened {len(sockets)} per-submesh sockets", flush=True)

        manifest = send_weights(mesh_device, hf_dict, sockets)
        print(
            f"[TTML rank {TTML_RANK}] sent {len(manifest['entries'])} tensors to {NUM_SUBMESHES} submeshes",
            flush=True,
        )

        # 1st barrier: source tensors stay alive until TTT has drained the
        # sockets. 2nd barrier: wait for TTT to finish verify + generate so
        # the two ranks tear down together (MPI would otherwise abort TTT
        # mid-generate when this rank exits).
        barrier("ttml", TTT_RANK)
        barrier("ttml", TTT_RANK)
    finally:
        sockets = None
        hf_dict = None
        model = None
        gc.collect()
        close_device()


# ---------------------------------------------------------------------------
# TTT rank
# ---------------------------------------------------------------------------


def _print_verify_report(report: List[dict], manifest: dict) -> None:
    total = len(report)
    passed = sum(1 for r in report if r["equal"])
    print("\n========= tensor-equality verification =========", flush=True)
    print(
        f"checked {total} (submesh, key) pairs across {manifest['num_submeshes']} submeshes "
        f"({len(manifest['entries'])} tensors each): {passed} equal, {total - passed} mismatched",
        flush=True,
    )
    fails = [r for r in report if not r["equal"]]
    if fails:
        for r in fails:
            print(f"  MISMATCH submesh={r['submesh']} key={r['key']}", flush=True)
    else:
        print("  all received tensors are torch.equal to the sender's reference", flush=True)
    print("================================================\n", flush=True)


def _ttt_side() -> None:
    from transformers import AutoTokenizer
    from utils.llama_ttt_presets import bf16_attn_bfp8_mlp_optimizations, llama_stop_and_pad
    from utils.ttt_generation_worker import TttGenerationWorker

    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()

    parent_mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*TTT_PARENT_MESH_SHAPE),
        offset=ttnn.MeshCoordinate(0, 0),
    )
    # One [1, 1] submesh per device. submeshes[i] is the chip at parent
    # coord (0, i), matching socket_i's TTML(0, i) -> submesh_i(0, 0) route.
    submeshes = parent_mesh.create_submeshes(ttnn.MeshShape(1, 1))
    assert len(submeshes) == NUM_SUBMESHES, f"expected {NUM_SUBMESHES} submeshes, got {len(submeshes)}"
    print(
        f"[TTT rank {TTT_RANK}] parent mesh {[int(d) for d in parent_mesh.shape]} -> {len(submeshes)} submeshes",
        flush=True,
    )

    workers: List[Any] = []
    sockets = None
    try:
        stop_token_ids, pad_token_id = llama_stop_and_pad(MODEL_ID)

        for i, submesh in enumerate(submeshes):
            print(f"[TTT rank {TTT_RANK}] building worker on submesh {i}", flush=True)
            workers.append(
                TttGenerationWorker(
                    mesh_device=submesh,
                    model_source=MODEL_ID,
                    max_batch_size=TTT_MAX_BATCH_SIZE,
                    max_seq_len=TTT_MAX_SEQ_LEN,
                    instruct=True,
                    optimizations=bf16_attn_bfp8_mlp_optimizations,
                    stop_token_ids=stop_token_ids,
                    pad_token_id=pad_token_id,
                    temperature=TEMPERATURE,
                    top_k=0,
                    top_p=1.0,
                    seed=0,
                )
            )

        sockets = open_receiver_sockets(submeshes)
        print(f"[TTT rank {TTT_RANK}] opened {len(sockets)} per-submesh sockets", flush=True)

        per_submesh, reference, manifest = recv_weights(submeshes, sockets)
        print(f"[TTT rank {TTT_RANK}] received {len(manifest['entries'])} tensors per submesh", flush=True)

        # Release the sender's source tensors now that everything is local.
        barrier("ttt", TTML_RANK)

        # ---- verification: exact tensor equality for everything sent ---- #
        report = verify_received(per_submesh, reference)
        _print_verify_report(report, manifest)

        # ---- apply + generate (printed, not asserted) ------------------- #
        for i, worker in enumerate(workers):
            worker.update_weights(per_submesh[i])
        print(f"[TTT rank {TTT_RANK}] applied weights to all {len(workers)} submesh models", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        prompt_ids = tokenizer.encode(PROMPT, add_special_tokens=True)

        print("\n========= per-submesh generation =========", flush=True)
        print(f"prompt: {PROMPT!r}", flush=True)
        for i, worker in enumerate(workers):
            out = worker.generate([prompt_ids], max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)[0]
            text = tokenizer.decode(out, skip_special_tokens=False)
            print(f"[submesh {i}] ({len(out)} tok) {text!r}", flush=True)
        print("==========================================\n", flush=True)

        # Final barrier: pairs with the sender's second barrier.
        barrier("ttt", TTML_RANK)
    finally:
        sockets = None
        workers = []
        gc.collect()
        ttnn.close_mesh_device(parent_mesh)


# Long, interactive, multi-model hardware test (HF download + four TTT
# worker builds before the transfer even starts). Disable the repo-wide
# 300s pytest-timeout default (pytest.ini) for this test specifically.
@pytest.mark.timeout(0)
def test_ddp_to_submesh_weight_transfer() -> None:
    if _MPI_RANK == TTML_RANK:
        _ttml_side()
    elif _MPI_RANK == TTT_RANK:
        _ttt_side()
    else:
        raise RuntimeError(
            f"Unexpected MPI rank {_MPI_RANK} (world_size={_WORLD_SIZE}); "
            f"expected exactly two ranks: TTML={TTML_RANK}, TTT={TTT_RANK}."
        )
