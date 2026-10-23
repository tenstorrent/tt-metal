# AutoDebug: Qwen3.8-27B HF output-gate contract

## Verdict

**No full-attention gate bug is established in the installed Transformers reference. Keep `torch.sigmoid(gate)`; do not replace it with SiLU/Swish. No implementation fix is needed for the reported discrepancy.**

The apparent contradiction comes from treating two different gates as one. The supported model contract uses **sigmoid for full-attention output gating**, and **SiLU/Swish for Gated DeltaNet (GDN) output gating after normalization**. In the inspected HF implementation, `output_gate_type` is retained as an extra config attribute but is not read by the Qwen3.5 modeling code. SGLang explicitly documents and consumes that field for the **linear-attention output norm**, which resolves its scope independently of its name. Its full-attention path still uses sigmoid.

This conclusion is supported by source inspection, not a numerical run of the checkpoint. It establishes the published inference/reference contract; it does not claim access to Qwen's private training implementation or prove end-to-end model accuracy.

## Scope and evidence identity

Investigation date: 2026-10-23. Repository HEAD: `0611d6e393882ce28e811b023b5f7d23bba1aced`.

- Inspected this autoport's reference artifacts, `/data/mvasiljev/weights/Qwen3.8-27B` config/model card/index and selected safetensors **headers**, and read-only source in container `qwen38-functional-build`.
- Container package metadata reports Transformers **5.12.1**, under `/opt/venv/lib/python3.10/site-packages/transformers/`.
- Inspected primary upstream HF and supported serving-framework sources. No other Qwen3.6/Qwen3.8 TT implementation was inspected. No model execution, tensor payload loading, device access, or implementation edit was performed.
- Qwen's published repository revision inspected was `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Its config is byte-identical to the local checkpoint config and [hf_config.json](hf_config.json). The model repository API lists no custom Python source; the config has no `auto_map` override.
- [hf_modeling_reference.py](hf_modeling_reference.py) is byte-identical to HF Transformers revision `e7b03417bc2bbf1dc2719d343813e229db3180aa`, `src/transformers/models/qwen3_5/modeling_qwen3_5.py`. It is **not** the same complete file as the installed 5.12.1 source. Their `Qwen3_5Attention` classes are nevertheless AST-identical, excluding source-location attributes.

SHA-256 identities:

| Artifact | SHA-256 |
| --- | --- |
| Local checkpoint config, saved config, and pinned published config | `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` |
| Saved HF modeling reference / pinned upstream source | `af05d0acfe79fff8aea1c75f93516a6a72bc80f6f901329a73e66a3866fe1e7c` |
| Installed 5.12.1 Qwen3.5 modeling source | `395439341ea5ba4dd14c103e830383caa75686cfaf6b693dd7efb58622224da6` |
| Installed 5.12.1 Qwen3.5 configuration source | `f9d6330bc658363330167f8a8943dff48f040566e1ab0f296eddb2ffb524d367` |

Config equality does not establish a content hash for the checkpoint's tensor payloads; those were not read.

## Direct observations

### 1. The owner publishes a Qwen3.5-family inference contract

The checkpoint declares `architectures = ["Qwen3_5ForConditionalGeneration"]`, outer `model_type = "qwen3_5"`, and text `model_type = "qwen3_5_text"`. It sets `attn_output_gate = true`, `output_gate_type = "swish"`, and `hidden_act = "silu"`. The model card identifies a hybrid stack of 48 GDN layers and 16 gated-attention layers and names Transformers, vLLM, and SGLang as compatible implementations. It does not instruct users to change HF attention to Swish. See the [pinned owner config](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json) and [pinned owner model card](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/README.md).

The `transformers_version = "5.8.0.dev0"` string is serialization metadata, not source evidence for an alternate gate equation.

### 2. HF preserves the config key but does not dispatch on it

Installed source locations below are relative to the container's Transformers package directory:

- `models/qwen3_5/configuration_qwen3_5.py:76-113`: `Qwen3_5TextConfig` has no declared `output_gate_type` field; its post-init forwards additional keyword arguments to the superclass.
- `configuration_utils.py:109-116`: additional configuration keys reach post-init.
- `configuration_utils.py:302-307`: remaining keys become attributes through `setattr`.
- Both installed and saved Qwen3.5 modeling files contain **zero occurrences** of `output_gate_type`.

Thus, “ignored by this forward implementation” is accurate; “dropped while loading config” is not. Setting the attribute again or changing its string cannot change this HF attention calculation. Likewise, HF always constructs the doubled query projection and output gate in this class; the checkpoint's true `attn_output_gate` value agrees with that behavior.

### 3. Full attention uses sigmoid, after attention and before `o_proj`

Installed `models/qwen3_5/modeling_qwen3_5.py:645-717` and saved `hf_modeling_reference.py:736-808` implement the same class. Relevant saved-source operations:

1. Lines 748-758: project hidden size 5120 to `24 * 256 * 2 = 12288` query-plus-gate elements; K and V each have `4 * 256 = 1024` elements.
2. Lines 774-777: view the query projection as `[batch, tokens, 24, 512]`, split the last dimension into a 256-element query and gate **within each head**, then flatten the gate to `[batch, tokens, 6144]`.
3. Lines 779-801: normalize Q/K, apply rotary embeddings, update K/V cache when present, and execute the selected attention backend. The gate does not pass through Q normalization or RoPE.
4. Lines 804-807: flatten attention output, multiply by `sigmoid(gate)`, then apply `o_proj` from 6144 to 5120.

The exact installed gate line is **714**; the saved-reference gate line is **805**. The multiplication is outside attention-backend dispatch, so selecting eager, SDPA, or another registered attention backend does not select a different gate activation here. Prefill and decode use this same post-attention expression.

Local safetensors headers for layer 3 corroborate the dimensions: `q_proj.weight=[12288,5120]`, `k_proj.weight=v_proj.weight=[1024,5120]`, and `o_proj.weight=[5120,6144]`, all BF16. These shapes establish the gate's width, not its activation.

### 4. GDN has a different output gate, and HF already uses SiLU there

Installed `Qwen3_5RMSNormGated`, lines 187-202, normalizes the recurrent output, applies its learned norm weight, multiplies by `F.silu(gate.float())`, and returns the input dtype. The saved source expresses the same activation through `self.activation = "silu"` and `ACT2FN[self.activation]`, lines 204-220.

In the saved GDN source:

- Lines 531 and 557-558: `in_proj_z` produces the separate GDN output-gate logits, reshaped into 48 value heads of width 128.
- Lines 643-649: recurrent output and `z` are flattened to rows of width 128, passed through `self.norm(core_attn_out, z)`, restored to width 6144, and passed through `out_proj`.
- Lines 605-607: the sigmoid on `b` controls the recurrence's write strength; the negative exponential/softplus expression controls decay. Neither is the full-attention gate nor the GDN output gate.

The layer-0 checkpoint headers confirm `in_proj_z.weight=[6144,5120]`, `norm.weight=[128]`, and `out_proj.weight=[5120,6144]`. SiLU and Swish with unit beta are the same function, `z * sigmoid(z)`.

### 5. A supported framework explicitly resolves the ambiguous field

At SGLang revision `0347b2bb4ef78a890c3359191b71161c867c1754`:

- [Qwen3NextConfig, lines 72-73](https://github.com/sgl-project/sglang/blob/0347b2bb4ef78a890c3359191b71161c867c1754/python/sglang/srt/configs/qwen3_next.py#L72) defines `output_gate_type` for the **linear-attention output norm**. [Qwen3_5TextConfig, line 15](https://github.com/sgl-project/sglang/blob/0347b2bb4ef78a890c3359191b71161c867c1754/python/sglang/srt/configs/qwen3_5.py#L15) inherits that config.
- [Qwen3.5 GDN, lines 393 and 507-518](https://github.com/sgl-project/sglang/blob/0347b2bb4ef78a890c3359191b71161c867c1754/python/sglang/srt/models/qwen3_5.py#L393) reads the field and supplies it to `RMSNormGated` with normalization before gating.
- The [gated-norm kernel, lines 184-196](https://github.com/sgl-project/sglang/blob/0347b2bb4ef78a890c3359191b71161c867c1754/python/sglang/kernels/ops/attention/fla/layernorm_gated.py#L184) maps both `swish` and `silu` to multiplication by `z * sigmoid(z)`.
- [Full attention, lines 1633-1644](https://github.com/sgl-project/sglang/blob/0347b2bb4ef78a890c3359191b71161c867c1754/python/sglang/srt/models/qwen3_5.py#L1633) separately invokes `fused_sigmoid_mul` before `o_proj`. Its [elementwise kernel, lines 35-41](https://github.com/sgl-project/sglang/blob/0347b2bb4ef78a890c3359191b71161c867c1754/python/sglang/kernels/ops/elementwise/elementwise.py#L35) multiplies attention output by sigmoid, without an extra factor of the gate logits.

This is stronger evidence than inferring meaning from the config key: the documented scope, actual caller, and kernel math agree. HF's hardcoded GDN SiLU matches this checkpoint's `swish` selection.

vLLM independently corroborates the full-attention equation at revision `7b1d07440dde8426a8b3d85e5f5815c06913eb82`: [Qwen3.5 layer construction, lines 157-166](https://github.com/vllm-project/vllm/blob/7b1d07440dde8426a8b3d85e5f5815c06913eb82/vllm/model_executor/models/qwen3_5.py#L157) selects `Qwen3NextAttention` for full attention; [that implementation, lines 463-468](https://github.com/vllm-project/vllm/blob/7b1d07440dde8426a8b3d85e5f5815c06913eb82/vllm/model_executor/models/qwen3_next.py#L463) applies sigmoid to the gate before `o_proj`.

## Causal adjudication

| Candidate explanation | Required fact | Inspection result | Decision |
| --- | --- | --- | --- |
| HF is wrong because the config demands Swish for full attention | The key must select the dense/full-attention gate | No such reader exists in the inspected HF class; SGLang scopes it to GDN and preserves dense sigmoid | Reject as a reference-fix rationale |
| Config loading loses the field | Loading must discard the extra key | Source preserves additional keys as attributes | Reject |
| Upstream main fixes a stale 5.12.1 dense gate | The installed and saved attention classes must differ semantically | Same AST, including projection, split, cache/backend calls, and sigmoid | Reject for this issue |
| Two distinct output gates explain the apparent mismatch | GDN must use Swish while full attention uses sigmoid | Confirmed in HF; explicitly configurable and independently confirmed in SGLang | Leading diagnosis, high source-level confidence |

No failing numerical output, comparison metric, or controlled passing/failing execution was supplied. Therefore there is no measured runtime failure to attribute to either activation. The concrete observation to explain is the config/source mismatch in the question, and the distinct gate scopes explain it.

Changing dense sigmoid to SiLU would introduce a real mathematical difference. With pre-gate attention value 1, the following ordinary scalar calculations hold:

| Gate logit | Existing sigmoid result | Proposed Swish result |
| --- | ---: | ---: |
| -1 | 0.268941 | -0.268941 |
| 0 | 0.5 | 0 |
| 1 | 0.731059 | 0.731059 |
| 2 | 0.880797 | 1.761594 |

The general pre-projection difference is `attention * sigmoid(gate) * (gate - 1)`. It is not rounding error or an algebraically equivalent implementation. The code does not constrain raw gate logits to 1. This is a symbolic counterexample, not a claim that those exact values occurred in the checkpoint.

The 16 potentially affected full-attention layers are zero-based indices `3,7,11,...,63`. If such a replacement were made, the first possible divergence from the unchanged full-model reference would be layer 3's post-attention gate, followed by `o_proj`, residual addition, and subsequent layers. GDN-only layer checks would not exercise that replacement. This is a predicted consequence of the proposed edit, not an observed failure in the existing implementation.

## Minimal intervention

**Leave the HF reference gate implementation unchanged.** Preserve sigmoid for full attention and SiLU/Swish for GDN's normalized output. Do not rewrite the checkpoint's `output_gate_type` to `sigmoid`, and do not add a global or model-local monkey patch that uses that field to choose the dense-attention activation.

For the functional-decoder work, the useful correction is to the reference specification: record the two gate contracts and pin the actual HF version/source used for golden outputs. This report provides that record. If a local reference was independently modified to use dense SiLU, reverting only that unsupported modification would be the minimal correction; no such modified reference was found in the artifacts inspected here.

Optional future CPU-only regression coverage can distinguish the gate equations with negative, zero, and positive logits, check per-head query/gate splitting, and exercise one GDN and one full-attention block. These are follow-up checks, not executions performed by this investigation. Hardware is unnecessary to establish this particular source contract, but full checkpoint correctness still needs the normal reference/TT validation outside this inspection-only task.

## Other potential issues and limits

- **Reference version drift:** the saved upstream file and installed package differ outside `Qwen3_5Attention`. The AST equality proves the dense-attention class comparison only. It is not certification that the two versions have identical GDN/cache/mask behavior or identical numerical outputs. Record which version executes a golden run rather than citing the other version's line numbers as its source.
- **General configuration coverage:** HF's lack of an `output_gate_type` reader means its hardcoded GDN activation would not honor a hypothetical checkpoint selecting a different GDN activation. This checkpoint selects `swish`, matching HF's SiLU, so that hypothetical limitation does not establish a bug here.
- **Porting check, not a discovered bug:** query and gate halves are paired inside each head's 512-wide projection slice. Splitting all 12288 features into two global 6144-wide halves would be a different layout. No implementation with that mistake was inspected or diagnosed.
- No claim is made about trained weight values, generated-text quality, TT kernels, or end-to-end accuracy. Private training semantics were not independently inspected. Model-owner compatibility declarations plus the matching supported inference paths establish the defensible reference contract available here.

## Verification and report review

Completed static checks: config JSON parsing and byte hashes; owner model-repository file listing; pinned-source byte comparison; AST parsing and attention-class comparison; manual config data-flow inspection; safetensors-header shape checks; layer-type enumeration; and scalar arithmetic separating sigmoid from Swish. No target model or device code ran.

The headline was challenged against the alternative that both HF versions share the same bug. Agreement between those versions alone would be insufficient. SGLang's explicit field documentation, GDN caller, gated-norm kernel, and separate full-attention kernel provide the additional evidence that rules out using this key as justification for a dense Swish patch. The remaining limits are recorded above rather than promoted to findings.

A fresh xhigh reviewer independently checked the first draft against the installed and saved source, config-retention path, artifact hashes, and SGLang field documentation/wiring. The review found no required corrections and agreed with the no-patch recommendation and its stated uncertainty limits.
