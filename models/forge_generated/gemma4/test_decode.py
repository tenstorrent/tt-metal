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
    n_slots = max(runtime) + 1
    input_list = [None] * n_slots
    for slot, t in runtime.items():
        input_list[slot] = t

    model = gemma4.Gemma4ForCausalLM.from_state_dict(
        hf,
        mesh_device,
        is_decode=True,
    )
    logits = model(input_list)

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
