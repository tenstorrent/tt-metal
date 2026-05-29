---
name: integration
description: Load real HuggingFace checkpoint weights into TTNN modules and validate full-config PCC against HF. Use when wiring real weights, weight loaders, or real-weight validation.
---

# SKILL: HF Weight Integration

## Purpose
Load real HuggingFace safetensors weights into TTNN module instances and
re-validate that each block hits PCC > 0.99 against the HF PyTorch
reference at production layer counts.

## When to use
- After the bringup phase produces TTNN modules with synthetic weights.
- Before any end-to-end demo or use-case validation.
- When a model's checkpoint is updated and per-block PCC needs to be re-verified.

## Prerequisites
- All blocks in `models/demos/{model}/tt/` instantiate and pass PCC > 0.99 with
  random weights against the matching `reference/functional.py` functions.
- A nested-state-dict contract per TTNN module is settled. The TTNN `__init__`
  signature is the source of truth — the loader must produce exactly that shape.
- The HF checkpoint is on disk locally (do not re-download per test).

## Process

### 1. Inspect the safetensors index

Read `model.safetensors.index.json` (or open each shard directly) to enumerate
the top-level prefixes. Map each prefix to one of the TTNN sub-models.

```python
from safetensors import safe_open
prefixes: dict[str, int] = {}
for shard in shards:
    with safe_open(shard, framework="pt") as f:
        for k in f.keys():
            top = k.split(".")[0]
            prefixes[top] = prefixes.get(top, 0) + 1
print(prefixes)  # e.g. {"text_encoder": 386, "text_decoder": 458, "vocoder": 401, ...}
```

Cross-reference with the component inventory in `ARCHITECTURE.md`. Every TTNN
sub-model must have exactly one HF prefix (or set of prefixes) assigned. Any
unassigned prefix means a missing component; any unmatched TTNN module means a
missing loader.

### 2. Write the weight loader (`tt/weight_loader.py`)

A single pure-PyTorch module that takes a flat HF `{key: tensor}` dict and
returns nested state_dicts shaped exactly as each TTNN `__init__` expects.
Rules:

- Pure PyTorch / safetensors only. No TTNN. No device touches. No I/O outside
  `load_hf_state_dict`.
- One function per block kind. Compose: `submodel_weights` calls
  `submodel_layer_weights(i)` in a list comprehension over `range(num_layers)`.
- Keep tiny helpers for the recurring `{"weight", "bias"}` shape:

```python
def _ln_sd(hf_sd, prefix):
    return {"weight": hf_sd[f"{prefix}.weight"], "bias": hf_sd[f"{prefix}.bias"]}

def _linear_sd(hf_sd, prefix, with_bias=True):
    sd = {"weight": hf_sd[f"{prefix}.weight"]}
    if with_bias and f"{prefix}.bias" in hf_sd:
        sd["bias"] = hf_sd[f"{prefix}.bias"]
    return sd
```

- Per-layer loader pattern:

```python
def encoder_layer_weights(hf_sd, layer_idx: int) -> dict:
    p = f"encoder.layers.{layer_idx}"
    return {
        "self_attn_layer_norm": _ln_sd(hf_sd, f"{p}.self_attn_layer_norm"),
        "self_attn": {
            "q_proj": _linear_sd(hf_sd, f"{p}.self_attn.q_proj"),
            "k_proj": _linear_sd(hf_sd, f"{p}.self_attn.k_proj"),
            "v_proj": _linear_sd(hf_sd, f"{p}.self_attn.v_proj"),
            "out_proj": _linear_sd(hf_sd, f"{p}.self_attn.out_proj"),
        },
        "ffn_layer_norm": _ln_sd(hf_sd, f"{p}.ffn_layer_norm"),
        "ffn": {
            "fc1": _linear_sd(hf_sd, f"{p}.ffn.fc1"),
            "fc2": _linear_sd(hf_sd, f"{p}.ffn.fc2"),
        },
    }
```

- Sub-model loader pattern (composes per-layer + module-level tensors):

```python
def text_encoder_weights(hf_sd, num_layers: int = 24) -> dict:
    return {
        "embed_tokens": {"weight": hf_sd["shared.weight"]},
        "embed_positions_weights": build_sinusoidal_positional_embedding_weights(...),
        "layers": [encoder_layer_weights(hf_sd, i) for i in range(num_layers)],
        "final_layer_norm": _ln_sd(hf_sd, "encoder.layer_norm"),
    }
```

- Expose `num_layers` as a keyword argument with the production default. The
  same loader then powers both the reduced-config harness (2 layers) and the
  full-config gate (24 layers) without code duplication.

- Add a `__main__` self-test that loads the checkpoint, calls every loader at
  full config, walks the returned tree counting tensor leaves, and spot-checks
  a few well-known shapes (e.g. `shared.weight` is `(vocab, hidden)`). This
  catches missing keys and wrong indexing before any TTNN code runs.

### 3. Handle the awkward cases

These are not edge cases — every nontrivial checkpoint hits at least one:

- **Weight tying.** When HF ties the token embedding to the LM head (and
  possibly to the decoder input embedding), the checkpoint stores ONE tensor
  (commonly `shared.weight`). Expose it through a single helper and let each
  sub-model loader embed it via that helper:

  ```python
  def shared_embedding_weight(hf_sd):
      return hf_sd["shared.weight"]

  def lm_head_weights(hf_sd):
      return {"weight": shared_embedding_weight(hf_sd)}  # tied; no bias
  ```

- **Buffers not in the checkpoint.** Sinusoidal positional embeddings,
  precomputed rotary tables, and ALiBi slopes are typically rebuilt at module
  init time and NOT serialized. Reconstruct them deterministically per the HF
  source. Port the constructor verbatim — do not "clean up" the math:

  ```python
  def build_sinusoidal_positional_embedding_weights(num_embeddings, embedding_dim, padding_idx=None):
      half_dim = embedding_dim // 2
      emb = math.log(10000) / (half_dim - 1)
      emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)
      emb = torch.arange(num_embeddings, dtype=torch.int64).float().unsqueeze(1) * emb.unsqueeze(0)
      emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
      if embedding_dim % 2 == 1:
          emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
      if padding_idx is not None:
          emb[padding_idx, :] = 0
      return emb.to(torch.get_default_dtype())
  ```

  Use the same `num_positions + offset` and `padding_idx` as the HF module
  (the offset is often a non-zero constant; grep the HF source for `offset =`).

- **Per-layer indexing.** HF stores everything flat as
  `submodel.layers.{i}.<attr>`. The TTNN modules expect either a nested
  per-layer list or a flat per-layer dict — match whichever the `__init__`
  takes. Never let layer indexing leak above the per-layer loader; the
  sub-model loader iterates `range(num_layers)` and that is the only place
  `i` appears.

- **Sibling vs nested fields.** Some HF modules pack auxiliary tensors
  (e.g. a relative-position `distance_embedding.weight`) inside an attention
  sub-module, but the TTNN block consumes them as a sibling kwarg of the
  enclosing layer. Expose them at the level the TTNN consumer expects, even
  if that re-shapes the HF nesting.

- **`bias=False` linears and convs.** Drop the bias key when the HF module
  was constructed without one. Guard with `if f"{prefix}.bias" in hf_sd` or
  pass `with_bias=False` explicitly — never insert a zero tensor as a stand-in.

### 4. Set up a parametric PCC test

One consolidated test file: `tests/test_real_hf_weights.py`. One pytest case
per block. Use parametrize so a single command runs the entire matrix.

```python
@pytest.fixture(scope="session")
def hf_sd():
    """Load the full HF checkpoint once per session."""
    return wl.load_hf_state_dict()

@pytest.fixture(scope="function")
def device():
    """Open a FRESH ttnn device per test.

    Sub-model instances allocate all per-layer weights into DRAM; with a
    session-scoped device those allocations accumulate and the largest blocks
    OOM near the end of the run. Per-test open/close costs ~1 s and gives
    clean isolation.
    """
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    yield dev
    ttnn.close_device(dev)

def _t_some_block(device, hf_sd) -> float:
    sd = wl.some_block_weights(hf_sd, layer_idx=0)
    x = build_input(...)
    ref_out = ref.some_block_forward(x, sd, ...)
    tt_block = SomeBlock(device=device, state_dict=sd, ...)
    tt_out = ttnn.to_torch(tt_block(_to_tt(device, x))).to(torch.float32)
    return _pcc_value(ref_out, tt_out.reshape(ref_out.shape))

BLOCKS = [
    ("layernorm", _t_layernorm),
    ("attention", _t_attention),
    # ... one row per block
]

@pytest.mark.parametrize("name,fn", BLOCKS, ids=[b[0] for b in BLOCKS])
def test_real_hf_weights(name, fn, device, hf_sd):
    pcc = fn(device, hf_sd)
    print(f"[{name}] real-HF PCC = {pcc:.6f}")
    assert pcc > 0.99, f"{name}: PCC {pcc} <= 0.99"
```

The session-scoped HF fixture amortizes the multi-second safetensors load
across the entire run. The function-scoped device fixture prevents DRAM
accumulation. Every per-block helper returns ONE float — no try/except, no
custom assertion logic — and the single `assert` lives in the parametric
wrapper.

### 5. The realistic-input trick

Random `N(0, 1)` inputs are out-of-distribution for trained weights and
produce activations that saturate bf16 precision. A trained encoder layer
typically multiplies a near-`N(0, 1)` input by an embedding scale (e.g.
`sqrt(hidden)` ≈ 32) and runs it through a LayerNorm before the first matmul.
Feeding raw `N(0, 1)` into an attention block bypasses that scaling and the
intermediate logits can blow up by ~600x, dropping bf16 PCC below 0.99 even
when the implementation is correct.

For any block that lives downstream of a token embedding (attention layers,
decoder layers, sub-models), build the input the same way the real model
does:

```python
def _build_realistic_post_ln_input(hf_sd, ln_prefix, batch=1, seq_len=64, embed_dim=1024, padding_idx=0):
    """Embed -> scale -> positional add -> LayerNorm. Matches what the
    block sees in production."""
    ln_sd = wl._ln_sd(hf_sd, ln_prefix)
    shared = wl.shared_embedding_weight(hf_sd)
    input_ids = torch.randint(low=2, high=512, size=(batch, seq_len), dtype=torch.long)
    pre_ln = F.embedding(input_ids, shared) * math.sqrt(embed_dim)
    sin_w = wl.build_sinusoidal_positional_embedding_weights(
        num_embeddings=max_pos + 2, embedding_dim=embed_dim, padding_idx=padding_idx
    )
    pre_ln = pre_ln + ref.sinusoidal_positional_embedding_forward(sin_w, input_ids=input_ids, padding_idx=padding_idx)
    return F.layer_norm(pre_ln, (embed_dim,), weight=ln_sd["weight"], bias=ln_sd["bias"])
```

When a block needs encoder hidden states for cross-attention, run a
reduced-config (2-layer) real reference encoder to produce them — do not feed
random tensors as K/V.

Use random inputs only for blocks whose first op is a `LayerNorm` /
`RMSNorm` (LN itself, FFNs that start with `gamma * normalize(x)`, vocoder
blocks that take spectrogram-scale inputs). For everything else, prefer
embed-derived input.

### 6. Two-stage validation

Run weight validation in two passes — the cheap one drives iteration, the
expensive one is the production gate:

**Stage 1 (`tests/test_real_hf_weights.py`, reduced config):** Per-block leaves
at full config; sub-models at `num_layers=2`. Target: PCC > 0.99 everywhere.
Runtime budget ~5 minutes. Use this as the inner loop while debugging weight
mapping issues.

**Stage 2 (`tests/test_full_config.py`, full config):** Same blocks, but
sub-models at the production layer count (e.g. 24 / 6). Target: PCC > 0.97 —
the deeper-stack threshold. Per-layer error is independent enough that
`0.999^24 ≈ 0.976`, so a 24-layer stack with PCC-0.999 leaves typically lands
near 0.97-0.98 even when every leaf is correct. Reserve full-config runs for
the final gate; expect them to take 10-30 minutes.

Keep sequence lengths SHORT in both stages (8-32 tokens). The goal is to
verify weight wiring and per-layer correctness compose — not to measure
latency. Latency is the perf phase's job.

If a block passes Stage 1 but fails Stage 2 see "Failure modes" below.

## Output artifacts

- `tt/weight_loader.py` — pure-PyTorch loader, one function per block.
- `tests/test_real_hf_weights.py` — Stage 1 parametric harness, all blocks
  PCC > 0.99.
- `tests/test_full_config.py` — Stage 2 full-config gate, sub-models
  PCC > 0.97.
- One row per block in the `BRINGUP_LOG.md` `real_weights` column, with the
  measured PCC.

## Failure modes

- **PCC drops at depth (Stage 2 fails after Stage 1 passes).** Per-layer
  error is compounding. Audit dtype assumptions in the per-layer chain
  (`fp32_dest_acc`, intermediate `bfloat8_b` matmul outputs, casts that
  silently snap to bf16). Re-run a chained per-layer PCC sweep — a single
  layer that drops from 0.9999 to 0.992 is the smoking gun. See
  `skills/debug/SKILL.md` (Mode B chained PCC) for the diagnostic procedure.

- **Missing checkpoint key.** Reverse-map from the missing attribute: which
  line of the HF `__init__` writes this `nn.Parameter`? If the answer is "the
  module rebuilds it from config" (sinusoidal embed, rotary table, ALiBi
  slope), add a builder in `weight_loader.py`. If the answer is "HF stores it
  under a different prefix" (tied weights, renamed modules), thread the
  correct key through.

- **HF module instantiation fails.** Wrong config keys. Cross-reference the
  HF source `__init__` against the `config.json` you loaded. Common traps:
  the model uses `hidden_size` but the config exposes `d_model`; the
  attention head count is per-group and you passed total heads.

- **Per-block PCC perfect but cross-attention sub-model fails.** The K/V
  source (encoder hidden states) is random rather than derived from a real
  reduced-config encoder. Wire a 2-layer encoder up to produce in-distribution
  K/V tensors before running the decoder block.

- **bf16 saturation on the first matmul.** Random `N(0, 1)` input is OOD.
  Apply the realistic-input pattern from step 5.

- **Shape mismatch on a Conv1d weight.** HF Conv1d weights are
  `[out_channels, in_channels, kernel_size]`. The TTNN consumer may expect a
  different layout. Match what the TTNN `__init__` takes — do not reshape in
  the loader unless the TTNN module documents that contract.

## Reference implementation

- `models/demos/facebook_seamless_m4t_v2_large/tt/weight_loader.py` — full
  worked example covering 24 block kinds across 5 sub-models, with shared
  embedding tying, sinusoidal rebuild, sibling-vs-nested handling, and
  bias=False convs.
- `models/demos/facebook_seamless_m4t_v2_large/tests/test_real_hf_weights.py`
  — Stage 1 parametric harness, 24 blocks, ~5 min runtime, PCC > 0.99.
- `models/demos/facebook_seamless_m4t_v2_large/tests/test_full_config.py` —
  Stage 2 full-config gate, sub-models at 24/6 layers, PCC > 0.97.
