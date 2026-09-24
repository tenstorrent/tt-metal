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

``--full`` is the whole shipped 48-layer stack on a real tokenized prompt, left-padded to 1024
with its attention mask — the exact input the pipeline builds. Two layers cannot show error that
only appears compounded over 48, and the other modes pass no mask at all, so this is the mode
that speaks to prompt adherence. Needs ~72 GB of RAM (fp32 + bf16 copies of 12B).
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


def full_config():
    """The shipped config as-is — all 48 layers."""
    text_config = dict(checkpoint_text_config())
    text_config.update(attn_implementation="eager", dtype="float32")
    return Gemma4TextConfig(**text_config)


def load_full_weights(model):
    with safe_open(CHECKPOINT, "pt") as handle:
        state = {
            key[len("model.") :]: handle.get_tensor(key).float() for key in handle.keys() if key.startswith("model.")
        }
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert all("v_proj" in k for k in missing), f"missing keys: {missing}"
    return model


def full_inputs(prompt: str, seq_len: int):
    """Mirror ``Gemma4TokenizerEncoderPair.tokenize``: manual BOS, then left-pad to seq_len."""
    import json as _json

    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    with safe_open(CHECKPOINT, "pt") as handle:
        tok_json = bytes(handle.get_tensor("tokenizer_json").numpy().tobytes())
        tok_cfg = _json.loads(bytes(handle.get_tensor("hf_asset__tokenizer_config.json").numpy().tobytes()))
    # transformers 5 reads model_max_length from the config itself, so passing it again collides.
    tok_cfg.pop("added_tokens_decoder", None)
    tok_cfg.pop("model_max_length", None)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_buffer(tok_json), model_max_length=seq_len, **tok_cfg
    )

    ids = tokenizer(prompt.strip(), padding=False, truncation=True, max_length=seq_len).input_ids
    if not ids or ids[0] != tokenizer.bos_token_id:
        ids = [tokenizer.bos_token_id, *ids][:seq_len]
    padded = tokenizer.pad(
        {"input_ids": [ids]}, padding="max_length", max_length=seq_len, return_tensors="pt", return_attention_mask=True
    )
    print(f"prompt tokens: {len(ids)} of {seq_len} (padding_side={tokenizer.padding_side})")
    return padded.input_ids, padded.attention_mask


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
    parser.add_argument("--full", action="store_true", help="all 48 shipped layers on a real prompt")
    parser.add_argument("--prompt-file", default=None, help="prompt for --full (default: a short T2V prompt)")
    parser.add_argument("--seq-len", type=int, default=1024, help="padded length for --full")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    suffix = "_full" if args.full else ("_real" if args.real else "")
    out = args.out or f"/tmp/g4ref/gemma4_reference{suffix}.safetensors"

    torch.manual_seed(0)
    config = full_config() if args.full else (real_config() if args.real else narrow_config())
    model = Gemma4TextModel(config).to(torch.float32).eval()

    if args.full:
        load_full_weights(model)
    elif args.real:
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

    attention_mask = None
    if args.full:
        prompt = (
            open(args.prompt_file).read()
            if args.prompt_file
            else "A young woman with shoulder-length wavy brown hair sits on a wooden stool, cradling an acoustic guitar."
        )
        input_ids, attention_mask = full_inputs(prompt, args.seq_len)
    else:
        input_ids = torch.randint(0, config.vocab_size, (1, SEQ_LEN))

    kwargs = {"output_hidden_states": True, "use_cache": False}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask

    with torch.no_grad():
        fp32 = model(input_ids=input_ids, **kwargs)
        # The same forward in bf16 gives the device test a floor to measure against: bf16
        # error compounds with depth, so an fp32-only target is unreachable and says nothing
        # about whether the port is correct.
        bf16 = model.to(torch.bfloat16)(input_ids=input_ids, **kwargs)

    tensors = {"input_ids": input_ids.to(torch.int64)}
    if attention_mask is not None:
        tensors["attention_mask"] = attention_mask.to(torch.int64)
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
