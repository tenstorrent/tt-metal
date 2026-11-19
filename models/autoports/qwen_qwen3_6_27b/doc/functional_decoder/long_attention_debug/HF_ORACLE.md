# HF long-attention oracle discriminator

**The tested HF oracle is bit identical to the installed HF decoder, including decode at context 262144.** These host controls refute the suspected omitted scale, causal-mask, softmax-backend, gate, and high-position conversion errors as explanations of the observed TT decode PCC 0.9563047. They do not establish the device-side cause.

The script `hf_oracle_check.py` extracts the actual nested `selected_forward` AST from `tests/run_long_context.py` and executes only that function with a CPU namespace. It never imports the harness, TTNN, or decoder runtime. It loads pinned real layer-3 weights for `Qwen/Qwen3.6-27B` revision `8f33b98937c6031bb49e713dee3e8e185aafd26f`, preserving hidden size 5120, 24 Q heads, 4 KV heads, head dimension 256 and the complete MLP. Torch is `2.11.0+cpu`. Source hash and all numerical results are in `hf_oracle_results.json`.

| Prefill length | Rotary positions | Native HF reference | Selected prefill | Decode |
|---:|---|---|---|---|
| 33 | 0–33 | Complete actual decoder prefill and DynamicCache decode | Bit identical | Bit identical, context 34 |
| 545 | 0–545 | Complete actual decoder prefill and DynamicCache decode | Bit identical | Bit identical, context 546 |
| 545 | 261598–262143 | Complete actual decoder prefill and DynamicCache decode | Bit identical | Bit identical at position 262143 |
| 262143 | 0–262143 | Actual decoder with all-token projected KV and selected causal mask; native DynamicCache decode | Bit identical | Bit identical, context 262144 |

Every output and contribution row PCC is 1.0; every maximum absolute error and changed-element count is zero; all values and correlations are finite. Selected rows include tile/chunk boundaries 31, 32, 511 and 512 where present and arbitrary late rows. At full context, the tested prefill rows are 0, 1, 31, 32, 511, 512, 262139, 262141 and 262142.

For the full-context control, every token's real normalized K/V projection and partial RoPE are evaluated in 512-token chunks once. The actual installed `Qwen3_5DecoderLayer` then evaluates selected prefill inputs through its own Q/gate projection, Q norm, RoPE, attention interface, output projection, residual and MLP. A read-only cache supplies the precomputed full K/V because a quadratic full-output CPU forward is unnecessary. The selected oracle sees the identical K/V. For decode, an actual `DynamicCache` holds the 262143-token prefix; the installed decoder projects and appends the new token itself. This checks the oracle's manual append against the native update. The input generator uses seed 291 and resets the RNG state immediately after real HF layer and rotary construction for each case. The original failed run did not save input hashes, so byte identity with that original run is not asserted.

The high-position short-context control offsets positions only at the rotary call. Causal indices remain local cache indices in both paths, preserving a correct causal mask while independently checking positional arithmetic near the advertised maximum. The full-context case requires no offset adapter.

Source correspondence:

- Installed `Qwen3_5Attention.scaling` is `256**-0.5 = 0.0625`, equal to the default scale in the oracle's PyTorch SDPA call. No additional model-specific scale is omitted.
- Both paths split each 512-wide Q projection head into 256-wide Q and gate, apply Q/K `(1+w)` RMSNorm, rotate the first 64 dimensions with the installed `apply_rotary_pos_emb`, and apply sigmoid to the unrotated gate after attention.
- The oracle's boolean mask permits K index `<=` absolute query index. HF's supplied boolean causal mask has the same meaning. Decode includes the new token and attends all prefix tokens plus itself.
- The installed HF SDPA wrapper passes explicit scaling and may repeat K/V when a mask is present; the oracle instead requests `enable_gqa=True`. These backend differences produced bit-identical results in all controls, including the full cache. Both use the installed PyTorch SDPA rather than an independent hand-coded BF16 softmax.
- `Qwen3_5TextRotaryEmbedding.forward` converts integer position IDs directly to FP32 before frequency multiplication and trigonometry, then converts cosine/sine outputs to the activation dtype. Position 262143 is exactly representable in FP32. Int64 and FP32 position inputs produce bit-identical RoPE in every case.

A negative control explicitly converts position 262143 to BF16 first: it rounds to 262144 and changes RoPE by maximum absolute 0.876953125. The actual harness does not perform that conversion. This distinguishes the real risk of BF16 position rounding from the checked oracle's behavior.

Limitations: the full-context prefill control shares the all-token K/V cache construction with the oracle. The small full-decoder controls independently validate that construction; the large control checks selected query/output behavior and full-length attention. This is a CPU oracle discriminator, not a TT correctness test or a full quadratic CPU output comparison. No TT source, hardware state or owner test harness was changed.

Reproduction (exit 0):

```bash
source python_env/bin/activate
source /home/ttuser/qwen36-27b/lane-env.sh
timeout 600 python models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/long_attention_debug/hf_oracle_check.py > models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/long_attention_debug/hf_oracle_run.log 2>&1
```

Artifacts: `hf_oracle_check.py`, `hf_oracle_results.json`, `hf_oracle_run.log`, and this report. The independent AutoDebug launcher and its artifacts are owned by the parent agent and were not modified here.
