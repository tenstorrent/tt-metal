"""Pinned text-only HF reference using the packaged readiness file format."""

import json
import time
from pathlib import Path

import torch
from readiness_check.generate import (
    DEFAULT_AIME24_PROMPTS_FILE,
    _generate_continuation_tokens,
    _generate_one_entry,
    _resolve_prompt_text,
)
from readiness_check.schema import Reference, save_reference
from transformers import AutoConfig, AutoTokenizer, GenerationConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

from models.autoports.google_gemma_4_31b_it.tt.weights import HF_MODEL, HF_REVISION, PinnedWeights


def load_reference(layer_indices=None):
    torch.set_num_threads(16)
    config = AutoConfig.from_pretrained(HF_MODEL, revision=HF_REVISION).text_config
    indices = list(range(config.num_hidden_layers)) if layer_indices is None else list(layer_indices)
    config.num_hidden_layers = len(indices)
    config.layer_types = [config.layer_types[i] for i in indices]
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = Gemma4ForCausalLM(config)
    loader = PinnedWeights()
    state = {}
    for target, source in enumerate(indices):
        print("HF_LOAD_LAYER", source, flush=True)
        state.update({f"model.layers.{target}.{k}": v for k, v in loader.layer(source).items()})
    terminal = loader.load(["model.language_model.embed_tokens.weight", "model.language_model.norm.weight"])
    state["model.embed_tokens.weight"] = terminal["model.language_model.embed_tokens.weight"]
    state["lm_head.weight"] = state["model.embed_tokens.weight"]
    state["model.norm.weight"] = terminal["model.language_model.norm.weight"]
    model.load_state_dict(state, assign=True, strict=True)
    # Nonpersistent rotary buffers constructed on meta need materialization too.
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    model.model.rotary_emb = Gemma4TextRotaryEmbedding(config)
    model.model.embed_tokens.embed_scale = torch.tensor(config.hidden_size**0.5)
    model.generation_config = GenerationConfig.from_pretrained(HF_MODEL, revision=HF_REVISION)
    return model.eval()


def main():
    start = time.perf_counter()
    root = Path("models/autoports/google_gemma_4_31b_it/doc/full_model")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, revision=HF_REVISION)
    prompt = _resolve_prompt_text(
        "aime24", prompt=None, prompt_file=None, aime24_prompts_file=DEFAULT_AIME24_PROMPTS_FILE, aime24_prompt_index=0
    )
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True, return_dict=False
    )
    root.joinpath("aime24_prompt.json").write_text(
        json.dumps(
            {
                "prompt": prompt,
                "tokens": tokens,
                "rendered": tokenizer.decode(tokens),
                "hf_model": HF_MODEL,
                "revision": HF_REVISION,
            },
            indent=2,
        )
        + "\n"
    )
    model = load_reference()
    inp = torch.tensor(tokens, dtype=torch.long)
    with torch.inference_mode():
        generated = _generate_continuation_tokens(
            model=model, tokenizer=tokenizer, prompt_tokens=inp, gen_len=100, device=torch.device("cpu")
        )
        entry = _generate_one_entry(
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=inp,
            gen_tokens=generated,
            top_k=100,
            device=torch.device("cpu"),
        )
    output = Path("bringup/references/google_gemma_4_31b_it_aime24_chat.refpt")
    save_reference(
        Reference(
            k=100,
            hf_model_id=HF_MODEL,
            entries=[entry],
            token_ids_meta={
                "bos_id": tokenizer.bos_token_id,
                "eos_id": tokenizer.eos_token_id,
                "pad_id": tokenizer.pad_token_id,
            },
        ),
        output,
    )
    meta = {
        "hf_model": HF_MODEL,
        "revision": HF_REVISION,
        "tokenizer": HF_MODEL,
        "tokenizer_revision": HF_REVISION,
        "prompt_source": "aime24",
        "chat_template": True,
        "generation_length": 100,
        "top_k": 100,
        "command": "python_env/bin/python -m models.autoports.google_gemma_4_31b_it.tests.generate_full_reference",
        "reference": str(output),
        "completion": tokenizer.decode(generated.tolist()),
        "tokens": generated.tolist(),
        "elapsed_seconds": time.perf_counter() - start,
    }
    root.joinpath("reference_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(meta["completion"], flush=True)


if __name__ == "__main__":
    main()
