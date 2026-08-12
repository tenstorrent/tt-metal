"""Generate a Gemma-4 reference forward pass for test_gemma4_parity.

Gemma-4 landed in transformers 5, but tt-metal's env pins 4.53 and upgrading it in place
would disturb every other model, so this runs out-of-process in a throwaway venv and hands
the result over as safetensors — which, unlike a pickle, does not care that the two sides
are on different torch versions::

    python -m venv /tmp/g4ref
    /tmp/g4ref/bin/pip install transformers==5.10.1
    /tmp/g4ref/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
    /tmp/g4ref/bin/python models/tt_dit/tests/encoders/gemma4/gen_gemma4_reference.py [--real]

Default mode is a narrow randomly-initialised stack, which isolates the arithmetic. ``--real``
takes two layers of trained weights at full width instead: one sliding and one global, since
those are the two layer shapes that exist.
"""

import argparse
import json
import struct

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

CHECKPOINT = (
    "/home/noblewoodall/.cache/ltx-checkpoints/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
)
SEQ_LEN = 128

# Layer 5 is the first global layer of the shipped 48-layer stack.
REAL_GLOBAL_LAYER = 5
REAL_SLIDING_LAYER = 0


def narrow_config():
    return Gemma4TextConfig(
        vocab_size=1000,
        hidden_size=1024,
        intermediate_size=2048,
        num_hidden_layers=6,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=256,
        global_head_dim=512,
        num_global_key_value_heads=1,
        attention_k_eq_v=True,
        sliding_window=1024,
        max_position_embeddings=SEQ_LEN,
        layer_types=["sliding_attention"] * 5 + ["full_attention"],
        rope_parameters={
            "full_attention": {"rope_type": "proportional", "rope_theta": 1000000.0, "partial_rotary_factor": 0.25},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        },
        rms_norm_eps=1e-6,
        attention_bias=False,
        hidden_activation="gelu_pytorch_tanh",
        num_kv_shared_layers=0,
        enable_moe_block=False,
        use_double_wide_mlp=False,
        hidden_size_per_layer_input=0,
        attn_implementation="eager",
    )


def checkpoint_text_config():
    with open(CHECKPOINT, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
    return json.loads(header["__metadata__"]["gemma_config"])["text_config"]


def real_config():
    """The shipped config, cut to one layer of each kind."""
    text_config = dict(checkpoint_text_config())
    text_config.update(
        num_hidden_layers=2,
        layer_types=["sliding_attention", "full_attention"],
        max_position_embeddings=SEQ_LEN,
        attn_implementation="eager",
        dtype="float32",
    )
    return Gemma4TextConfig(**text_config)


def load_real_weights(model):
    """Map the two chosen checkpoint layers onto the two-layer stack."""
    remap = {"embed_tokens.weight": "model.embed_tokens.weight", "norm.weight": "model.norm.weight"}
    with safe_open(CHECKPOINT, "pt") as handle:
        available = set(handle.keys())
        for dst, src in ((0, REAL_SLIDING_LAYER), (1, REAL_GLOBAL_LAYER)):
            prefix = f"model.layers.{src}."
            for key in (k for k in available if k.startswith(prefix)):
                remap[f"layers.{dst}." + key[len(prefix) :]] = key

        state = {dst: handle.get_tensor(src).float() for dst, src in remap.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected}"
    # v_proj is absent by design on the global layer, where V is tied to K.
    assert all("v_proj" in k for k in missing), f"missing keys: {missing}"
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="use trained weights at full width")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out = args.out or (f"/tmp/g4ref/gemma4_reference{'_real' if args.real else ''}.safetensors")

    torch.manual_seed(0)
    config = real_config() if args.real else narrow_config()
    model = Gemma4TextModel(config).to(torch.float32).eval()

    if args.real:
        load_real_weights(model)
    else:
        # Distinct per-layer scalars so a dropped or misplaced multiply cannot pass.
        for idx, layer in enumerate(model.layers):
            layer.layer_scalar.fill_(1.0 + 0.05 * (idx + 1))

    for idx, layer in enumerate(model.layers):
        attn = layer.self_attn
        print(
            f"layer {idx}: {attn.layer_type:18} head_dim={attn.head_dim:4} v_proj={'None' if attn.v_proj is None else 'yes'}"
        )

    input_ids = torch.randint(0, config.vocab_size, (1, SEQ_LEN))

    with torch.no_grad():
        fp32 = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        # The same forward in bf16 gives the device test a floor to measure against: bf16
        # error compounds with depth, so an fp32-only target is unreachable and says nothing
        # about whether the port is correct.
        bf16 = model.to(torch.bfloat16)(input_ids=input_ids, output_hidden_states=True, use_cache=False)

    tensors = {"input_ids": input_ids.to(torch.int64)}
    # hidden_states[i] for i < N is the *input* to layer i, and the last entry is already
    # the post-norm output — the final layer's pre-norm activation is never exposed.
    for i, (a, b) in enumerate(zip(fp32.hidden_states, bf16.hidden_states)):
        tensors[f"hidden.{i}"] = a.contiguous().clone().float()
        tensors[f"bf16.{i}"] = b.contiguous().clone().float()

    if not args.real:
        # Real weights stay in the checkpoint so the device side exercises loading them.
        tensors.update({f"weight.{k}": v.float().contiguous() for k, v in model.float().state_dict().items()})

    save_file(tensors, out)
    print(f"\nhidden_states: {len(fp32.hidden_states)} (last is post-norm)")
    print(
        f"last_hidden_state {tuple(fp32.last_hidden_state.shape)} "
        f"mean={fp32.last_hidden_state.mean():.5f} std={fp32.last_hidden_state.std():.5f}"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
