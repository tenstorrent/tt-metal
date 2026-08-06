# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hardware PCC gate for the on-device HunyuanVideo-1.5 byT5 (glyph) encoder.

NOT YET RUN. Everything in this file opens a Tenstorrent mesh; the host-only
evidence lives in `test_byt5_encoder_host.py`. Run these on an idle machine
before considering `HY_TT_BYT5=1` for anything beyond experiments:

    HF_HUB_OFFLINE=1 pytest -svv \\
        models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_byt5_encoder_pcc.py

byT5 tensor parallelism must divide both `num_heads` (6) and `d_model` (1472),
so only 1- and 2-device meshes are legal; both are parametrized here. The
encoder is tiny (12 layers over 256 tokens), so these cases are quick.
"""

import glob
import os

import pytest
import torch

import ttnn
from models.demos.hf_eager.hunyuanvideo_1_5.tt.byt5_encoder import DEFAULT_PROMPT_LENGTH, TTByT5EncoderAdapter
from models.demos.hf_eager.hunyuanvideo_1_5.tt.pipeline import HunyuanVideo15Pipeline

_SNAPSHOT_GLOB = "models--hunyuanvideo-community--HunyuanVideo-1.5-Diffusers-480p_*"

# Two quoted spans, i.e. exactly the shape of prompt that actually reaches byT5.
_GLYPH_PROMPT = 'a neon sign reading "OPEN" beside another reading "24/7"'


def _snapshot_subdir(name):
    hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    matches = sorted(glob.glob(os.path.join(hub, _SNAPSHOT_GLOB, "snapshots", "*", name)))
    return matches[0] if matches else None


def _pcc(a, b):
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denominator = a.norm() * b.norm()
    return 0.0 if denominator == 0 else float(torch.dot(a, b) / denominator)


def _load_reference():
    encoder_dir = _snapshot_subdir("text_encoder_2")
    tokenizer_dir = _snapshot_subdir("tokenizer_2")
    if not encoder_dir or not tokenizer_dir:
        pytest.skip("no local HunyuanVideo-1.5 snapshot with text_encoder_2 + tokenizer_2")

    from diffusers.pipelines.hunyuan_video1_5.pipeline_hunyuan_video1_5 import extract_glyph_texts
    from transformers import AutoTokenizer, T5EncoderModel

    host = T5EncoderModel.from_pretrained(encoder_dir, local_files_only=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    glyph = extract_glyph_texts(_GLYPH_PROMPT)
    assert glyph is not None, "the test prompt must contain quoted glyph text"
    tokens = tokenizer(
        glyph,
        padding="max_length",
        max_length=DEFAULT_PROMPT_LENGTH,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        # Exactly the call `HunyuanVideo15Pipeline._get_byt5_prompt_embeds` makes.
        reference = host(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask.float())[0]
    return host, tokens.input_ids, tokens.attention_mask, reference


@pytest.mark.parametrize("mesh_device", [(1, 1), (1, 2)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 8192, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_byt5_real_weight_pcc(mesh_device):
    """Valid-token PCC against the host byT5 on a real glyph prompt."""
    host, ids, mask, reference = _load_reference()

    # The built-in first-call self-check is disabled so this test owns the
    # thresholds and can report both the valid and the padding correlation.
    adapter = TTByT5EncoderAdapter(host, mesh_device, verify=False, zero_padding=True)
    try:
        actual = adapter(input_ids=ids, attention_mask=mask.float())[0]
    finally:
        adapter.deallocate_weights()

    assert actual.shape == reference.shape
    valid = mask.bool().unsqueeze(-1).expand_as(reference)
    valid_pcc = _pcc(reference[valid], actual[valid])
    padding_pcc = _pcc(reference[~valid], actual[~valid])
    print(
        f"byT5 {tuple(mesh_device.shape)} valid PCC={valid_pcc:.6f} "
        f"({int(mask.sum())} tokens); padding PCC={padding_pcc:.6f}",
        flush=True,
    )
    assert valid_pcc >= 0.99

    # Padding positions are deliberately zeroed on device and are undefined on
    # the host, but the DiT only ever sees the trimmed prefix -- compare that.
    reference_trimmed, _ = HunyuanVideo15Pipeline._trim_to_valid(reference, mask)
    actual_trimmed, _ = HunyuanVideo15Pipeline._trim_to_valid(actual, mask)
    assert _pcc(reference_trimmed, actual_trimmed) >= 0.99


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 8192, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_byt5_full_sequence_pcc_without_padding_neutralization(mesh_device):
    """With `zero_padding=False` the adapter must reproduce HF everywhere.

    This is the strict parity case: if it passes, the on-device padding embeds
    are as good as the valid ones and `HY_BYT5_ZERO_PAD` is only a defensive
    measure. If it fails while `test_byt5_real_weight_pcc` passes, padding
    neutralization is load bearing and must stay on.
    """
    host, ids, mask, reference = _load_reference()
    adapter = TTByT5EncoderAdapter(host, mesh_device, verify=False, zero_padding=False)
    try:
        actual = adapter(input_ids=ids, attention_mask=mask.float())[0]
    finally:
        adapter.deallocate_weights()

    full_pcc = _pcc(reference, actual)
    print(f"byT5 full-sequence PCC (no padding neutralization) = {full_pcc:.6f}", flush=True)
    assert full_pcc >= 0.99


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 8192, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_byt5_batched_glyph_rows_match_per_row_encoding(mesh_device):
    """Two prompts in one batch must give the same result as encoding each alone.

    The mask enters attention as a per-row additive bias, so an incorrectly
    broadcast mask would show up here and nowhere else.
    """
    host, ids, mask, _ = _load_reference()
    short_ids = ids.clone()
    short_mask = mask.clone()
    keep = max(1, int(mask.sum()) // 2)
    short_ids[:, keep:] = 0
    short_mask[:, keep:] = 0

    batched_ids = torch.cat([ids, short_ids], dim=0)
    batched_mask = torch.cat([mask, short_mask], dim=0)

    adapter = TTByT5EncoderAdapter(host, mesh_device, verify=False, zero_padding=True)
    try:
        batched = adapter(input_ids=batched_ids, attention_mask=batched_mask.float())[0]
        rows = [
            adapter(input_ids=batched_ids[i : i + 1], attention_mask=batched_mask[i : i + 1].float())[0]
            for i in range(2)
        ]
    finally:
        adapter.deallocate_weights()

    for i, row in enumerate(rows):
        pcc = _pcc(row, batched[i : i + 1])
        print(f"byT5 batched row {i} PCC vs standalone = {pcc:.6f}", flush=True)
        assert pcc >= 0.999


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 8192, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_byt5_builtin_self_check_passes_on_the_production_path(mesh_device):
    """The adapter's own fail-closed first-call check must not trip."""
    host, ids, mask, _ = _load_reference()
    adapter = TTByT5EncoderAdapter(host, mesh_device, verify=True, pcc_threshold=0.99)
    try:
        adapter(input_ids=ids, attention_mask=mask.float())
    finally:
        adapter.deallocate_weights()
