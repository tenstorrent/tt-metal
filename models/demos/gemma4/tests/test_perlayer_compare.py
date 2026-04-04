"""Per-layer comparison of TT vs HF with CORRECT (single) embedding scaling."""
import os

import torch

import ttnn

os.environ.setdefault("USER", "node")
os.environ.setdefault("LOGNAME", "node")
import getpass

getpass.getuser = lambda: "node"

HF_MODEL = os.environ.get(
    "HF_MODEL",
    "/workspace/group/gemma4_weights/models--google--gemma-4-E4B-it/snapshots/292a7e278a400932df35f9fd4b1501edd04133a5",
)


def run_comparison(mesh_device):
    from transformers import AutoModelForImageTextToText

    from models.common.utility_functions import comp_pcc

    # Load HF model
    print("Loading HF model...")
    hf_model = AutoModelForImageTextToText.from_pretrained(HF_MODEL, dtype=torch.bfloat16)
    hf_model.eval()
    text_model = hf_model.model.language_model

    # Disable per-layer gating
    for layer in text_model.layers:
        layer.hidden_size_per_layer_input = 0
    text_model.hidden_size_per_layer_input = 0
    if hasattr(text_model, "config"):
        text_model.config.hidden_size_per_layer_input = 0
    if hasattr(hf_model.model, "config"):
        hf_model.model.config.text_config.hidden_size_per_layer_input = 0
    hf_model.config.text_config.hidden_size_per_layer_input = 0

    # KV sharing is ENABLED (essential for correct model behavior, TT model now supports it)

    input_ids = torch.tensor([[818, 5279, 529, 7001, 563]])

    # Capture per-layer outputs using hooks
    hf_layer_outputs = {}
    hf_embed_output = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            hf_layer_outputs[layer_idx] = output[0].detach().clone()

        return hook

    def embed_hook(module, input, output):
        hf_embed_output["embed"] = output.detach().clone()

    # Register hooks
    text_model.embed_tokens.register_forward_hook(embed_hook)
    for i, layer in enumerate(text_model.layers):
        layer.register_forward_hook(make_hook(i))

    # Run HF full forward (with correct single scaling)
    print("Running HF forward...")
    with torch.no_grad():
        outputs = hf_model(input_ids=input_ids)

    hf_embed = hf_embed_output["embed"]
    print(f"HF embed (single-scaled): shape={hf_embed.shape}, mag={hf_embed[0,:5,:].float().norm(dim=-1).mean():.4f}")

    hf_logits = outputs.logits[0, -1, :].float()
    print(f"HF logits: min={hf_logits.min():.4f}, max={hf_logits.max():.4f}")
    top5_hf = torch.topk(hf_logits, 5)
    print(f"HF top-5: {top5_hf.indices.tolist()} vals: {[f'{v:.2f}' for v in top5_hf.values.tolist()]}")

    # Print HF layer magnitudes
    print("\nHF per-layer magnitudes:")
    for i in sorted(hf_layer_outputs.keys()):
        if i < 8 or i in [11, 17, 23, 29, 35, 41]:
            h = hf_layer_outputs[i]
            print(f"  L{i:2d} shape={h.shape} ndim={h.ndim}")
            if h.ndim == 3:
                mag = h[0, :5, :].float().norm(dim=-1).mean().item()
            elif h.ndim == 2:
                mag = h[:5, :].float().norm(dim=-1).mean().item()
            else:
                mag = h.float().norm().item()
            is_global = i in [5, 11, 17, 23, 29, 35, 41]
            print(f"  L{i:2d} ({'GLOBAL' if is_global else 'slide '}) mag={mag:.4f}")

    # Now run TT model
    print("\n\nLoading TT model...")
    from models.demos.gemma4.tt.gemma4_model import TtGemma4TextModel
    from models.demos.gemma4.tt.model_config import ModelArgs
    from models.tt_transformers.tt.common import Mode

    model_args = ModelArgs(
        mesh_device=mesh_device,
        instruct=True,
        dummy_weights=False,
        max_batch_size=1,
        max_seq_len=128,
    )
    state_dict = model_args.load_state_dict()
    model_dtype = ttnn.bfloat16
    weight_cache_path = model_args.weight_cache_path(model_dtype)

    model = TtGemma4TextModel(
        args=model_args,
        dtype=model_dtype,
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
    )

    # Prepare inputs
    tokens = torch.tensor([[818, 5279, 529, 7001, 563]], dtype=torch.long)
    padded_tokens = torch.nn.functional.pad(tokens, (0, 128 - 5), value=0)

    prefill_inputs = model.prepare_inputs_prefill(padded_tokens, start_pos=0, last_token_idx=4)
    (tt_tokens_embd, rot_mats_global, rot_mats_local, tt_page_table, tt_chunk_page_table) = prefill_inputs

    # Check embedding
    tt_embd = ttnn.to_torch(ttnn.from_device(tt_tokens_embd))[0, 0, :5, :].float()
    hf_embd = hf_embed[0, :5, :].float()
    _, embd_pcc = comp_pcc(hf_embd.unsqueeze(0), tt_embd.unsqueeze(0))
    print(f"\nEmbedding PCC: {embd_pcc:.6f}")
    print(f"  TT mag={tt_embd.norm(dim=-1).mean():.4f}, HF mag={hf_embd.norm(dim=-1).mean():.4f}")

    # Run TT layer by layer
    x = tt_tokens_embd
    print("\nPer-layer comparison (TT vs HF):")
    print(f"{'Layer':>5} {'Type':>6} {'TT mag':>8} {'HF mag':>8} {'Ratio':>7} {'PCC':>10}")
    print("-" * 55)

    # KV sharing setup: track source layer K/V
    shared_kv_store = {}
    kv_source_layers = set()
    if hasattr(model_args, "kv_sharing_map"):
        for src_idx in model_args.kv_sharing_map.values():
            kv_source_layers.add(src_idx)

    for i, layer in enumerate(model.layers):
        from models.tt_transformers.tt.model_config import TensorGroup

        activation_dtype = model_args.decoders_optimizations.get_tensor_dtype(
            decoder_id=i, tensor=TensorGroup.ACTIVATION
        )
        if activation_dtype is not None and x.dtype != activation_dtype:
            x = ttnn.typecast(x, activation_dtype)

        # Set shared K/V for KV-shared layers
        if hasattr(model_args, "kv_sharing_map") and i in model_args.kv_sharing_map:
            src_idx = model_args.kv_sharing_map[i]
            if src_idx in shared_kv_store:
                layer.attention.shared_kv = shared_kv_store[src_idx]

        x = layer(
            x,
            current_pos=None,
            rot_mats_global=rot_mats_global,
            rot_mats_local=rot_mats_local,
            user_id=0,
            mode=Mode.PREFILL,
            page_table=tt_page_table,
        )

        # Store K/V from source layers
        if i in kv_source_layers:
            shared_kv_store[i] = (layer.attention.last_k_heads, layer.attention.last_v_heads)

        tt_out = ttnn.to_torch(ttnn.from_device(x))[0, 0, :5, :].float()
        h = hf_layer_outputs[i]
        hf_out = (h[0, :5, :] if h.ndim == 3 else h[:5, :]).float()
        _, pcc = comp_pcc(hf_out.unsqueeze(0), tt_out.unsqueeze(0))
        tt_mag = tt_out.norm(dim=-1).mean().item()
        hf_mag = hf_out.norm(dim=-1).mean().item()
        ratio = tt_mag / hf_mag if hf_mag > 0 else float("inf")
        is_global = i in [5, 11, 17, 23, 29, 35, 41]
        ltype = "GLOBAL" if is_global else "slide"
        print(f"  L{i:2d} {ltype:>6} {tt_mag:8.4f} {hf_mag:8.4f} {ratio:7.4f} {pcc:10.6f}")

    print("\nDone!")


if __name__ == "__main__":
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        run_comparison(mesh_device)
    finally:
        ttnn.close_mesh_device(mesh_device)
