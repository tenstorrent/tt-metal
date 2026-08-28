import pytest
import torch

import ttnn
from models.demos.flux_2_klein_9b.text_encoder.tt import model_ref
from models.demos.flux_2_klein_9b.text_encoder.tt.pipeline import build_pipeline, graduated_invocation_probe


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [8], indirect=True)
def test_smoke(mesh_device):
    hf = model_ref.load_hf_model(torch.float32)
    p = build_pipeline(mesh_device, model=hf, layers=2)
    ids = model_ref.encode_prompt(model_ref.DEFAULT_PROMPT)
    print("[smoke] prompt_len", ids.shape, flush=True)

    with graduated_invocation_probe() as c:
        emb = p.run_prompt_encoding(input_ids=ids)
    print("[smoke] prompt_embeds", tuple(emb.shape), "counts", dict(c), flush=True)

    # reference with the same 2-layer truncation

    ref_model = hf.model
    keep = ref_model.layers
    ref_model.layers = torch.nn.ModuleList(list(keep)[:2])
    with torch.no_grad():
        gold = ref_model(inputs_embeds=ref_model.embed_tokens(ids)).last_hidden_state.float()
    ref_model.layers = keep
    print("[smoke] call2 PCC", model_ref.pcc(gold, emb), flush=True)

    with graduated_invocation_probe() as c2:
        gen = p.run_text_generation(input_ids=ids, max_new_tokens=3)
    print(
        "[smoke] gen ids", gen["token_ids"], "logits", tuple(gen["step_logits"].shape), "counts", dict(c2), flush=True
    )
    print("[smoke] text", repr(gen["text"]), flush=True)
