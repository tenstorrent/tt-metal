# Lowering Spec: LM Head And Sampling Boundary

Source: `Gemma4ForConditionalGeneration.forward`, Transformers `v5.5.0`, `modeling_gemma4.py` lines 2400-2451.

## Contract

Input hidden state shape is `[1, 1, S, 2816]`. Output logits are `[1, 1, S, 262144]` logically.

The checkpoint ties `lm_head.weight` to `model.language_model.embed_tokens.weight`. The TTNN path materializes an LM head view as `[2816, 262144]`.

## Pseudocode

```python
logits = hidden_states @ embed_tokens.weight.T
logits = tanh(logits / 30.0) * 30.0
next_token = sample_or_argmax(logits)
```

## TTNN Decomposition

1. Keep LM head weights BF16 for correctness-first quality checks.
2. TP-shard vocab dimension across the 8-chip mesh.
3. Apply softcap before any gather or sampling.
4. For decode, prefer on-device sampling over reading full vocab logits back to host.
5. If host sampling is used for debug, gather logits and record that it is not the accepted traced decode path.

## Edge Cases

The per-device vocab width must be padded to a power of two for efficient TTNN top-k sampling. The logical output must still map back to the 262144-token tokenizer vocabulary.
