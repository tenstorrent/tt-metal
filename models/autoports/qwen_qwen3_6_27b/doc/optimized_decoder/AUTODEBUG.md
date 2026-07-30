# AutoDebug: optimized full-attention real-weight PCC

Scope: source-only investigation. The repo-local AutoDebug runner was invoked,
but its nested sandbox could not create a `bwrap` namespace; the investigation
therefore continued in a fresh unsandboxed Codex context. No TT hardware was
used and the supplied failing command was not run:
`timeout 900 python models/autoports/qwen_qwen3_6_27b/tests/full_attention_real_pcc.py --candidate default`.

## Direct observations

1. The failing test is a strict layer-3 official-weight HF-vs-TTNN one-token
   decode check. It loads only `model.language_model.layers.3.*` tensors from
   the checkpoint index (`tests/full_attention_real_pcc.py:32-44`), constructs
   a strict HF `Qwen3_5DecoderLayer` (`:47-54`), uses a random BF16
   `[1, 1, 5120]` hidden state at position 0 (`:67-77`), then compares the TTNN
   output against HF at PCC 0.995 (`:207-222`). The test path is valid for
   catching an official-weight attention-layer mismatch.

2. Current source makes `--candidate default` ambiguous for full attention.
   `POLICIES["default"]` is the aggressive BFP4/LoFi policy
   (`tt/optimized_decoder.py:50-58`), but `OptimizedDecoder.from_state_dict`
   calls `resolve_policy` (`:323`) and `resolve_policy("default",
   "full_attention")` returns `POLICIES["bf16_attention_bfp4_mlp_bfp8_cache"]`
   (`:207-214`, `:136-145`). The real PCC test prints only the user-supplied
   candidate string (`tests/full_attention_real_pcc.py:216-220`), so a log line
   saying `candidate=default` does not identify the effective dtype/fidelity
   policy in current source.

3. HF's `q_proj` order is per-head interleaved query/gate, not a flat
   query-half then gate-half. HF defines `q_proj` with output width
   `num_attention_heads * head_dim * 2` (`modeling_qwen3_5.py:658-660`) and
   splits it by first viewing the output as `[..., heads, head_dim * 2]`, then
   chunking the last dimension (`:681-687`). Current TT code instead does a flat
   split: functional decode uses `q_and_gate[..., :q_width]` and
   `q_and_gate[..., q_width:]` (`tt/functional_decoder.py:721-724`), and
   optimized materialization does the same on the transposed host weight before
   packing (`tt/optimized_decoder.py:477-483`). A CPU-only official-weight probe
   with the same seed found PCC `0.023247115` for flat query vs HF query and
   `0.038692862` for flat gate vs HF gate.

4. The supplied failure value has the signature of a missing/corrupted attention
   branch, not just the flat gate split. A CPU-only one-token layer-3 calculation
   with official weights gave PCC `0.076941140` when the attention contribution
   was suppressed before the post-attention RMSNorm/MLP, essentially matching
   the supplied `0.07804663948285913`. The same CPU calculation using the
   current flat gate split but otherwise preserving the attention value path gave
   PCC `0.686675251`, which is bad but not close to the observed near-zero PCC.

5. Current optimized decode has an explicit QKV-head-helper layout mitigation.
   `_decode_linear` returns an L1 width-sharded output (`tt/optimized_decoder.py:
   539-560`). In `_full_attention_decode`, current source now comments that
   passing that width-sharded QKV tensor directly to
   `nlp_create_qkv_heads_decode` silently assigns wrong values to heads for
   dense real weights, with diagonal synthetic weights masking it, and converts
   `qkv` to `ttnn.L1_MEMORY_CONFIG` first (`:703-710`). Focused evidence supplied
   with this investigation confirms the boundary: packed projection and split
   PCC were both `0.999841`, but V-head PCC after
   `nlp_create_qkv_heads_decode` on the width-sharded input was `-0.02028`.
   Converting QKV to L1 interleaved before head creation raised V-head PCC to
   `0.999837`. This is the direct cause of the original near-zero result, and
   current source contains its narrow fix.

6. The synthetic full-attention fixture is not discriminating for these bugs.
   It uses diagonal projection weights (`tests/full_attention_synthetic_pcc.py:
   29-47`); the functional docs record full-attention synthetic decode PCC
   `0.999960815` but no full-attention official-weight decode gate
   (`doc/functional_decoder/README.md:59-63`), and
   `context_contract.json` explicitly marks full-attention real-weight decode as
   false (`doc/context_contract.json:49-64`). A CPU-only diagonal-q-proj probe
   still showed flat-vs-HF q/gate mismatch, but the final layer PCC can remain
   high because the fixture and one-token residual path mask the semantic error.

7. The built-in `--probe-qkv` path does not validate HF q/gate ordering. Its
   expected packed tensor is built with the same flat q/gate split as the
   optimized code (`tests/full_attention_real_pcc.py:146-164`), and its split
   checks compare only against that self-consistent expectation (`:166-186`).
   It can validate the optimized matmul against its own packed host tensor, but
   it cannot refute the HF order mismatch above.

8. Other reference conventions checked out. HF RMSNorm is an offset parameter:
   the module initializes weight to zeros (`modeling_qwen3_5.py:737-742`) and
   multiplies by `1.0 + weight` (`:746-751`). Functional and optimized loaders
   both add one during materialization (`tt/functional_decoder.py:214-223`;
   `tt/optimized_decoder.py:362-368`, `:400-413`). Layer kind and shapes also
   match the target contract: layer 3 is a `full_attention` representative in
   config/docs (`doc/context_contract.json:89-99`).

9. The remaining precision boundary is independently measured. After the
   QKV-layout fix, BFP8 attention reached only `0.898606` against the functional
   official-weight output. BF16 attention plus BFP4 MLP reached `0.997086`, and
   retaining a BFP8 cache reached `0.997073`. Thus BF16/HiFi4 attention is
   required for the optimized-vs-functional stage-preservation gate, while a
   BFP8 cache is acceptable for this test.

10. The functional decoder is not an HF-correct oracle for full attention:
    supplied official-weight evidence gives functional-vs-HF PCC `0.687928`,
    consistent with the shared flat q/gate split. Optimized-vs-functional
    `~0.997` therefore proves preservation of the current stage, not HF
    correctness. The direct HF threshold in `full_attention_real_pcc.py` cannot
    pass until the shared q/gate convention is corrected.

## Interpretations

The strongest source-backed issue that remains in current code is the
`q_proj` q/gate ordering bug shared by functional and optimized full-attention
paths. This is a real HF convention mismatch and would be invisible to
optimized-vs-functional comparisons because both paths split the tensor the
same wrong way; the measured functional-vs-HF PCC `0.687928` confirms its
practical impact.

The original `0.078046...` is explained by a separate, now-mitigated layout
contract violation: the packed matmul was correct, but the head-creation helper
misread its width-sharded result. The V-head A/B (`-0.02028` before interleaving,
`0.999837` after) proves this earlier divergence. After that fix, BF16 attention
is still needed to preserve the functional stage; cache precision is not the
limiting axis in the supplied A/B.

Cache length, causal masking, and RoPE are lower-probability for this specific
failure. The failing test decodes one token into an empty cache at position 0,
so softmax has one key and Q/K/RoPE cannot change the attention value selection.
The value projection, gate, cache write/read, head concat, and output projection
are the meaningful attention-path surfaces for this test.

## Ranked hypotheses

1. **Confirmed cause of the reported `0.078046...`: QKV head creation consumed
   a width-sharded matmul result with the wrong layout.** The packed projection
   and split were correct (`0.999841`), V heads were wrong (`-0.02028`), and the
   current L1-interleaving boundary (`tt/optimized_decoder.py:703-710`) restores
   V heads to `0.999837`. Suppressing attention on CPU independently gives final
   PCC `0.076941140`, matching the supplied symptom.

2. **Definitely present in current source: wrong HF q/gate extraction from
   `self_attn.q_proj`.** Evidence: HF per-head split at
   `modeling_qwen3_5.py:681-687` conflicts with TT flat splits in
   `tt/functional_decoder.py:721-724` and `tt/optimized_decoder.py:477-483`.
   CPU official-weight flat-vs-HF PCC is near zero for both q and gate. Causal
   fit: affects real dense weights and is masked by the diagonal synthetic
   fixture; for the one-token failure it mainly corrupts the gate.

3. **Confirmed stage-preservation requirement: BFP8 attention is still too
   lossy after the layout fix.** BFP8 attention gives only `0.898606` against
   functional; BF16 attention plus BFP4 MLP gives `0.997086`, and changing the
   cache to BFP8 retains `0.997073`. Current full-attention default resolution
   correctly selects the BF16-attention policy (`tt/optimized_decoder.py:
   136-145`, `:207-214`).

4. **Test/probe ambiguity can hide the true axis.** Evidence: the test prints
   `candidate=default` instead of the resolved policy
   (`tests/full_attention_real_pcc.py:216-220`), and `--probe-qkv` uses the same
   flat packed expectation as optimized (`:146-164`). Causal fit: a passing
   probe would not prove HF correctness, and a failure log cannot tell whether
   the aggressive or resolved full-attention policy was used.

5. **Lower probability for this command: page table, current position, causal
   mask, or RoPE.** Evidence: the test uses position 0, page table `[[0]]`, and
   a fresh cache (`tests/full_attention_real_pcc.py:67-77`, `:126-137`);
   one-token attention makes Q/K/RoPE irrelevant to value selection. These still
   need later coverage after the basic attention-value/gate path is fixed.

## Focused verify/refute experiments

1. After correcting q/gate ordering in both decoders, rerun the direct HF test
   and the optimized-vs-functional preservation test separately. Expected:
   functional-vs-HF and optimized-vs-HF both exceed `0.995`, while
   optimized-vs-functional remains at least the measured `~0.997`.

2. Extend the probe locally to compare `q_proj` output against the HF per-head
   extraction, not the current flat expectation: reshape the projected
   `q_and_gate` as `[heads, 2 * head_dim]`, split the last dimension, and compare
   those q/gate tensors with the TT tensors used before gate multiplication.

3. Preserve the completed precision A/B as a regression: explicit
   `--candidate bf16_hifi4`, `--candidate bf16_attention_bfp4_mlp_bfp8_cache`,
   and `--candidate bfp8_hifi2`; print the resolved dtype/fidelity policy, not
   only the candidate alias.

4. Replace the diagonal synthetic full-attention fixture with a deterministic
   dense nonzero projection fixture for `q_proj`, `k_proj`, `v_proj`, and
   `o_proj`. Keep a dedicated sub-probe for q/gate extraction so synthetic PCC
   cannot pass solely through residual dominance.

## Fix direction, not applied

Repack `self_attn.q_proj.weight` according to HF order during loader setup:
after transposing to `[hidden, 2 * q_width]`, reshape the output-column axis as
`[num_heads, 2 * head_dim]`, take `[..., :head_dim]` for q and
`[..., head_dim:]` for gate, then flatten each back to `[hidden, q_width]`.
Use that q in `[q, k, v, gate]` packing and apply the same convention in the
functional path or make the optimized test compare directly to HF.

Keep the current L1 conversion before `nlp_create_qkv_heads_decode`; the supplied
V-head A/B directly verifies that intervention boundary. Keep BF16/HiFi4
attention as the full-attention default while allowing BFP4 MLP and BFP8 cache,
which met the stage-preservation gate.
