"""PCC gate for the prefill side.

Builds Gemma4ForCausalLM via the HF-only Phase 3 path
(build_cached_main_from_hf) + synthesized runtime inputs, runs the
forward pass, and asserts PCC >= 0.99 against the reference logits at
gemma4/reference_logits/prefill.pt.
"""
import pathlib

import pytest
import torch
import ttnn

import gemma4
from gemma4 import weights as gw


_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def mesh_device():
    """Open the (1,4) mesh device via the shared singleton."""
    return gemma4.utils.DeviceGetter.get_device((1, 4))


def test_prefill_pcc(mesh_device):
    """End-to-end prefill PCC gate. Uses HF weights + synthesized
    runtime inputs + skipping consteval entirely."""
    hf = gw.load_hf_weights()
    runtime = gemma4.synthesize_prefill_inputs(mesh_device)
    layer_table = gemma4.LAYER_TABLE_PREFILL

    max_norm_slot = max(
        max(t.get("q_norm_input", 0), t.get("k_norm_input", 0))
        for t in layer_table.values()
    )
    n_slots = max(max(runtime) + 1, max_norm_slot + 1)
    input_list = [None] * n_slots
    for slot, t in runtime.items():
        input_list[slot] = t

    cached_main = gw.build_cached_main_from_hf(
        hf, input_list, layer_table, mesh_device, is_decode=False,
    )

    model = gemma4.Gemma4ForCausalLM.from_consteval(
        layer_table=layer_table, is_decode=False,
    )
    out = model(input_list, cached_main=cached_main)
    logits = out[-1]

    logits_host = ttnn.from_device(logits)
    logits_torch = ttnn.to_torch(
        logits_host,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )[:1, :, :].float()

    ref = torch.load(
        _REPO_ROOT / "gemma4" / "reference_logits" / "prefill.pt",
        weights_only=True,
    ).float()
    pcc = torch.corrcoef(
        torch.stack([ref.flatten(), logits_torch.flatten()])
    )[0, 1].item()
    last_tok = logits_torch.shape[1] - 1
    print(f"prefill PCC = {pcc:.6f} | argmax: "
          f"ref={ref[0, last_tok].argmax().item()}, "
          f"out={logits_torch[0, last_tok].argmax().item()}")
    assert pcc > 0.99, f"PCC dropped: {pcc}"
