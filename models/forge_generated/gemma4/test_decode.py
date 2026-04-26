"""PCC gate for the decode side. Same shape as test_prefill.py.

Builds Gemma4ForCausalLM via the HF state_dict path + synthesized
runtime inputs, runs the forward pass, and asserts PCC >= 0.99 against
the reference logits at gemma4/reference_logits/decode.pt.
"""
import pathlib

import gemma4
import pytest
import torch
from gemma4 import weights as gw
from gemma4.layer_table import LAYER_TABLE_DECODE

import ttnn

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def mesh_device():
    """Open the (1,4) mesh device via the shared singleton."""
    return gemma4.utils.DeviceGetter.get_device((1, 4))


def test_decode_pcc(mesh_device):
    """End-to-end decode PCC gate. Uses HF weights + synthesized
    runtime inputs."""
    hf = gw.load_hf_weights()
    runtime = gemma4.synthesize_decode_inputs(mesh_device)
    layer_table = LAYER_TABLE_DECODE

    max_norm_slot = max(max(t.get("q_norm_input", 0), t.get("k_norm_input", 0)) for t in layer_table.values())
    n_slots = max(max(runtime) + 1, max_norm_slot + 1)
    input_list = [None] * n_slots
    for slot, t in runtime.items():
        input_list[slot] = t

    model = gemma4.Gemma4ForCausalLM.from_state_dict(
        hf,
        mesh_device,
        is_decode=True,
    )
    out = model(input_list)
    logits = out[-1]

    logits_host = ttnn.from_device(logits)
    logits_torch = ttnn.to_torch(
        logits_host,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )[:1, :, :].float()

    ref = torch.load(
        _REPO_ROOT / "gemma4" / "reference_logits" / "decode.pt",
        weights_only=True,
    ).float()
    pcc = torch.corrcoef(torch.stack([ref.flatten(), logits_torch.flatten()]))[0, 1].item()
    last_tok = logits_torch.shape[1] - 1
    print(
        f"decode PCC = {pcc:.6f} | argmax: "
        f"ref={ref[0, last_tok].argmax().item()}, "
        f"out={logits_torch[0, last_tok].argmax().item()}"
    )
    assert pcc > 0.99, f"PCC dropped: {pcc}"
