# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""``DeepSeekV41DSpark`` (the checkpoint's ``mtp.0..2`` block drafter) against ``Transformer.forward_spec``.

The drafter runs TP4 on a 1x4 submesh of an 8x4 Galaxy, on its own: random target-layer hiddens stand
in for the decode model's stash. A case seeds the rings with ``prompt`` positions, then drafts at the
next few positions, writing each position's ring row first, as the reference's ``forward_spec`` does.
Each draft compares the pre-Markov logits by PCC, the proposals and the confidence logits.

The routed experts come from the cache ``tools/fill_expert_cache.py --dspark`` fills.

    pytest models/experimental/deepseek_v41_flash/tests/decode/test_dspark.py
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.deepseek_v41_flash.tests.reference import load_reference, reference_dspark
from models.experimental.deepseek_v41_flash.tt.config import CACHE_DIR, DEFAULT_MODEL_DIR, dequantized, load_config
from models.experimental.deepseek_v41_flash.tt.decode.dspark import DeepSeekV41DSpark, host_to_device
from models.experimental.deepseek_v4_flash.tt.layers import Linear
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader, resolve_snapshot_dir

# Stage 0 attention sees only the reference's own inputs and caches, so it is held tight at every frontier.
ATTENTION_PCC = 0.999
# A near-tie in a stage's top-3-of-128 routing (the scores are bf16) can flip and send a row to other experts;
# random hidden states make such ties common. The MoE is judged on the device's own input, by the median, and
# the end-to-end drift only loosely.
MOE_PCC = 0.999
MEAN_LOGITS_PCC = 0.93
MIN_FIRST_ID_MATCH = 0.5
TP_SIZE = 4
PARENT_MESH = (8, 4)
DRAFTS = 8


def _snapshot():
    try:
        return resolve_snapshot_dir(DEFAULT_MODEL_DIR)
    except FileNotFoundError:
        return None


def _to_torch(t: ttnn.Tensor) -> torch.Tensor:
    """Rank 0's copy of a tensor replicated over the mesh."""
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def _keep_output(outputs: dict, key):
    """Forward hook storing the module's output in ``outputs[key]``."""

    def hook(module, args, out):
        outputs[key] = out

    return hook


def _spy(obj, method: str, outputs: dict, key) -> None:
    """Have ``obj.<method>`` also store DRAM copies of its first argument and its output (the first, for
    a tuple) in ``outputs[key]``, as ``(input, output)``."""
    call = getattr(obj, method)

    def copy(t):
        return ttnn.clone(ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG))

    def spied(*args, **kwargs):
        given = copy(args[0])
        out = call(*args, **kwargs)
        outputs[key] = (given, copy(out[0] if isinstance(out, tuple) else out))
        return out

    setattr(obj, method, spied)


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}], indirect=True)
@pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True, ids=["8x4"])
@pytest.mark.parametrize("prompt", (10, 140), ids=("first_lap", "wrapped"))
@torch.no_grad()
def test_dspark_draft(mesh_device, prompt):
    if tuple(mesh_device.shape) != PARENT_MESH:
        pytest.skip(f"needs an {PARENT_MESH[0]}x{PARENT_MESH[1]} mesh, got {tuple(mesh_device.shape)}")
    device = mesh_device.create_submesh(ttnn.MeshShape(1, TP_SIZE), ttnn.MeshCoordinate(0, 0))
    snapshot = _snapshot()
    config = load_config(snapshot)
    loader = DeepseekV4WeightLoader(snapshot)
    max_seq = prompt + DRAFTS + 16
    model = load_reference(snapshot)
    reference = reference_dspark(model, snapshot, loader, max_seq)
    freqs = reference.mtp[0].attn.freqs_cis

    cache = WeightCache(CACHE_DIR)
    lm_head = Linear(
        dequantized(loader, "lm_head.weight", translate=True), device, cache.file("lm_head"), dtype=ttnn.bfloat16
    )
    drafter = DeepSeekV41DSpark(
        config,
        loader,
        device,
        lm_head,
        (freqs.real, freqs.imag),
        cache=cache,
        # The drafter's math, without most of bfloat4_b's routing flips (bfloat8_b fits on a bare device).
        expert_dtype=ttnn.bfloat8_b,
        tp_size=TP_SIZE,
    )
    logits_seen = {}

    def lm_head_spy(x):
        logits_seen["logits"] = lm_head(x)
        return logits_seen["logits"]

    drafter.lm_head = lm_head_spy
    # Each stage's attention and MoE outputs and output streams, device and reference, for the log.
    got_parts, want_parts = {}, {}
    for s, (stage, block) in enumerate(zip(drafter.stages, reference.mtp)):
        for name in ("attention", "moe", "forward"):
            _spy(stage, name, got_parts, (s, name))
        block.attn.register_forward_hook(_keep_output(want_parts, (s, "attention")))
        block.ffn.register_forward_hook(_keep_output(want_parts, (s, "moe")))
        block.register_forward_hook(_keep_output(want_parts, (s, "forward")))

    torch.manual_seed(0)
    hidden, n_targets = config.hidden_size, len(drafter.targets)
    main_hidden = torch.randn(1, max_seq, n_targets * hidden).bfloat16().float()
    tokens = torch.randint(config.vocab_size, (max_seq,))
    stash = torch.zeros(n_targets, 1, 1, drafter.stash_rows, hidden)
    for p in range(max_seq):
        stash[:, 0, 0, p % drafter.stash_rows] = main_hidden[0, p].reshape(n_targets, hidden)
    for k, pieces in enumerate(drafter.stash):
        for j, dst in enumerate(pieces):
            w = dst.shape[-1]
            host_to_device(stash[k][..., j * w : (j + 1) * w], dst)

    reference.spec(tokens[:1], main_hidden[:, :prompt], 0)
    seed = list(range(max(0, prompt - drafter.window), prompt))
    for i in range(0, len(seed), drafter.main_rows):
        chunk = seed[i : i + drafter.main_rows]
        drafter.set_main_rows(chunk, [True] * len(chunk))
        drafter.write_main()

    failures, first_matches, pccs, moe_alone = [], 0, [], []
    for f in range(prompt, prompt + DRAFTS):
        token = int(tokens[f + 1])
        want_ids, want_logits, want_conf = reference.spec(tokens[f + 1 : f + 2], main_hidden[:, f : f + 1], f)
        markov = torch.stack([reference.mtp[-1].markov_head(want_ids[:, i])[0] for i in range(drafter.block)], dim=1)
        drafter.set_main_rows([f], [True])
        drafter.write_main()
        drafter.set_block(token, f)
        ids, conf = drafter.draft()
        got_ids = _to_torch(ids).flatten().long()
        got_conf = _to_torch(conf).float().flatten()
        got_logits = _to_torch(logits_seen["logits"]).float().reshape(drafter.block, -1)
        want_logits = (want_logits - markov).reshape(drafter.block, -1)
        want_ids, want_conf = want_ids[0, 1:], want_conf.flatten()

        for (s, name), want in sorted(want_parts.items()):
            want = want[0] if isinstance(want, tuple) else want
            given, got = (_to_torch(t).float() for t in got_parts[(s, name)])
            _, pcc = comp_pcc(want.float(), got.reshape(want.shape))
            report = f"frontier {f}: stage {s} {name}: PCC {pcc}"
            if (s, name) == (0, "attention") and pcc < ATTENTION_PCC:
                failures.append(f"frontier {f}: stage 0 attention PCC {pcc} < {ATTENTION_PCC}")
            if name == "moe":
                # The MoE alone: the reference on the device's own input.
                _, alone = comp_pcc(reference.mtp[s].ffn(given.reshape(want.shape)), got.reshape(want.shape))
                moe_alone.append(alone)
                report += f" (given the device's input {alone})"
            logger.info(report)
        _, pcc = comp_pcc(want_logits, got_logits)
        pccs.append(pcc)
        first_matches += int(got_ids[0] == want_ids[0])
        logger.info(
            f"frontier {f}: logits PCC {pcc}, ids {got_ids.tolist()} vs {want_ids.tolist()}, "
            f"confidence {got_conf.tolist()} vs {want_conf.tolist()}"
        )
    mean_pcc = sum(pccs) / len(pccs)
    median_moe = sorted(moe_alone)[len(moe_alone) // 2]
    logger.info(
        f"mean logits PCC {mean_pcc}, first proposal matched {first_matches} / {DRAFTS}, "
        f"median MoE PCC on the device's input {median_moe}"
    )
    if median_moe < MOE_PCC:
        failures.append(f"median MoE PCC on the device's input {median_moe} < {MOE_PCC}")
    if mean_pcc < MEAN_LOGITS_PCC:
        failures.append(f"mean logits PCC {mean_pcc} < {MEAN_LOGITS_PCC}")
    if first_matches < MIN_FIRST_ID_MATCH * DRAFTS:
        failures.append(f"first proposal matched {first_matches} / {DRAFTS} drafts")
    assert not failures, "; ".join(failures)
