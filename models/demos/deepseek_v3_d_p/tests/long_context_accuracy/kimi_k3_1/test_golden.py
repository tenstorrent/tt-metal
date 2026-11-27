# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host contracts of the K3.1 FP64 chain-golden reader (``KimiK31ChainGolden``) on a small registered chain: rows across
segment and file boundaries, sampled rows below a bound, refusal of products or rows the golden does not hold and of a
missing or other-version chain, and the per-layer weight set and digest. CPU only."""

from __future__ import annotations

import json

import torch
from safetensors.torch import save_file

from models.demos.common.prefill.long_context_accuracy.fp64_golden import (
    GoldenMismatch,
    GoldenNotFound,
    product_schema,
    write_manifest,
    write_token_ids,
)
from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3_1 import golden as chain

ROWS, SEGMENT = 10, 4  # segments [0, 4) [4, 8) [8, 10)


def _write_chain(root, golden_id="k31-test-chain", producer_version=chain.PRODUCER_VERSION):
    """A registered chain golden: row r's ``moe_output_1`` is ``[r, -r]``, its key ``latent`` ``[10 r]``, and every
    segment samples its first row (``attn_output_3`` = ``[r + 0.5]``)."""
    directory = root / golden_id
    directory.mkdir(parents=True)
    segments = []
    for i, start in enumerate(range(0, ROWS, SEGMENT)):
        r = torch.arange(start, min(start + SEGMENT, ROWS), dtype=torch.float32)
        payload = {
            "start": start,
            "end": start + r.numel(),
            "moe_output_1": torch.stack([r, -r], 1),
            "sampled_rows": torch.tensor([start]),
            "attn_output_3": torch.tensor([[start + 0.5]], dtype=torch.float64),
        }
        state = {"keys": {"latent": (10 * r).double().unsqueeze(1)}, "kda": {}}
        torch.save(payload, directory / f"segment_{i:06d}.pt")
        torch.save(state, directory / f"state_{i:06d}.pt")
        segments.append({"start": start, "end": start + r.numel()})
    digest = write_token_ids(directory, torch.arange(ROWS))
    products = {
        "moe_output_1": product_schema("row", payload["moe_output_1"]),
        "latent": product_schema("row", state["keys"]["latent"]),
        "sampled_rows": product_schema("sampled_row", payload["sampled_rows"]),
        "attn_output_3": product_schema("sampled_row", payload["attn_output_3"]),
    }
    write_manifest(
        directory,
        {
            "golden_id": golden_id,
            "producer": chain.PRODUCER,
            "producer_version": producer_version,
            "identity": {"model": chain.MODEL, "token_source": "test", "weights_sha256": {}},
            "token_ids_sha256": digest,
            "rows": ROWS,
            "segment": SEGMENT,
            "segments": segments,
            "products": products,
        },
    )


def test_read_rows_spans_segments_and_files(tmp_path) -> None:
    _write_chain(tmp_path)
    golden = chain.open_k31_chain("k31-test-chain", tmp_path)
    got = golden.read_rows(["moe_output_1", "latent"], 3, 9)
    assert torch.equal(got["moe_output_1"], torch.tensor([[3.0, -3], [4, -4], [5, -5], [6, -6], [7, -7], [8, -8]]))
    assert torch.equal(got["latent"], torch.tensor([[30.0], [40], [50], [60], [70], [80]], dtype=torch.float64))
    assert torch.equal(golden.read_rows(["moe_output_1"], 9, 10)["moe_output_1"], torch.tensor([[9.0, -9]]))


def test_read_rows_refuses_what_the_golden_does_not_hold(tmp_path, expect_error) -> None:
    _write_chain(tmp_path)
    golden = chain.open_k31_chain("k31-test-chain", tmp_path)
    with expect_error(ValueError, "sampled-row products"):
        golden.read_rows(["attn_output_3"], 0, 4)
    with expect_error(ValueError, "has no products"):
        golden.read_rows(["moe_output_2"], 0, 4)
    with expect_error(ValueError, "not inside"):
        golden.read_rows(["moe_output_1"], 8, 11)


def test_read_sampled_keeps_rows_below_end(tmp_path) -> None:
    _write_chain(tmp_path)
    golden = chain.open_k31_chain("k31-test-chain", tmp_path)
    every = golden.read_sampled()
    assert every["sampled_rows"].tolist() == [0, 4, 8]
    assert every["attn_output_3"].flatten().tolist() == [0.5, 4.5, 8.5]
    assert golden.read_sampled(8)["sampled_rows"].tolist() == [0, 4]  # (row 8 is not below 8)


def test_open_fails_on_a_missing_or_other_version_chain(tmp_path, expect_error) -> None:
    with expect_error(GoldenNotFound, "no golden"):
        chain.open_k31_chain("k31-test-chain", tmp_path)
    _write_chain(tmp_path, producer_version=chain.PRODUCER_VERSION + 1)
    with expect_error(GoldenMismatch, "producer_version"):
        chain.open_k31_chain("k31-test-chain", tmp_path)


ATTN_RES = [
    "input_layernorm.weight",
    "mlp_res_norm.weight",
    "mlp_res_proj.weight",
    "post_attention_layernorm.weight",
    "self_attention_res_norm.weight",
    "self_attention_res_proj.weight",
]
ROUTER = ["block_sparse_moe.gate.weight", "block_sparse_moe.gate.e_score_correction_bias"]
DSA = [
    f"self_attn.{n}"
    for n in (
        "q_a_proj.weight q_a_layernorm.weight q_b_proj.weight kv_a_proj_with_mqa.weight kv_a_layernorm.weight "
        "kv_b_proj.weight g_proj.weight o_proj.weight indexer.wq_b.weight indexer.wk.weight indexer.k_norm.weight "
        "indexer.k_norm.bias indexer.weights_proj.weight"
    ).split()
]


def test_layer_weights_reads_the_chain_tensor_set(tmp_path, expect_error) -> None:
    """Layer 3 (DSA): the AttnRes, router and DSA weights, under the checkpoint's key root, digested by name; layer 0
    (dense FFN) is not supported."""
    names = ATTN_RES + ROUTER + DSA
    assert len(set(names)) == len(names) == 6 + 2 + 13
    assert sorted(chain.layer_weight_names(tmp_path, 3)) == sorted(names)
    tensors = {f"model.layers.3.{n}": torch.full((2,), float(i), dtype=torch.bfloat16) for i, n in enumerate(names)}
    tensors["model.layers.4.input_layernorm.weight"] = torch.zeros(2)  # (another layer's: not read)
    save_file(tensors, tmp_path / "shard.safetensors")
    weight_map = {k: "shard.safetensors" for k in tensors}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    got, digest = chain.layer_weights(tmp_path, 3)
    assert sorted(got) == sorted(names)
    assert torch.equal(got["self_attn.o_proj.weight"], tensors["model.layers.3.self_attn.o_proj.weight"])
    assert digest == {"layers.3": chain.tensor_digest(got)}
    got["self_attn.o_proj.weight"] = got["self_attn.o_proj.weight"] + 1
    assert chain.tensor_digest(got) != digest["layers.3"]
    with expect_error(ValueError, "not layer 0"):
        chain.layer_weights(tmp_path, 0)


def test_layer_weights_of_a_moe_layer_add_kda_and_the_non_routed_moe(tmp_path) -> None:
    """Layers 1 and 2 (KDA + MoE): the AttnRes weights, the KDA weights of the checkpoint's config (here a low-rank
    gate), the router and the MoE's shared expert and latent pair; not the routed experts."""
    linear = {"head_dim": 4, "num_heads": 2, "short_conv_kernel_size": 4}
    config = {"hidden_size": 8, "rms_norm_eps": 1e-6, "linear_attn_config": linear}
    (tmp_path / "config.json").write_text(json.dumps({"text_config": config}))
    kda = (
        "q_proj.weight k_proj.weight v_proj.weight q_conv1d.weight k_conv1d.weight v_conv1d.weight A_log "
        "f_a_proj.weight f_b_proj.weight dt_bias b_proj.weight o_norm.weight o_proj.weight g_a_proj.weight "
        "g_b_proj.weight"
    ).split()
    moe = (
        "routed_expert_down_proj routed_expert_up_proj routed_expert_norm shared_experts.gate_proj "
        "shared_experts.up_proj shared_experts.down_proj"
    ).split()
    want = [
        *ATTN_RES,
        *(f"self_attn.{n}" for n in kda),
        "block_sparse_moe.gate.weight",
        "block_sparse_moe.gate.e_score_correction_bias",
        *(f"block_sparse_moe.{n}.weight" for n in moe),
    ]
    assert len(want) == 6 + 15 + 2 + 6
    assert chain.layer_weight_names(tmp_path, 1) == want
    assert chain.layer_weight_names(tmp_path, 2) == want
