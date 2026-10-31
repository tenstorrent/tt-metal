# Greedy sampling contract

Both controls return maximum-valued tokens at B1/32/16/8 across unique winners, cross-shard ties, many ties and negative logits. Physical split top32 with semantic k1,p0,temp1 is compared against correctly row-padded force-argmax. The96 synthetic acceptance records all pass greedy semantics; ties may choose a different maximum index from torch.argmax. These are warmed sampler-only traced host times with128 repeats, not full-model or device times.

| Batch | Split greedy ms | Force argmax ms |
|---:|---:|---:|
| 1 | 0.4822 | 2.8944 |
| 32 | 0.5010 | 2.8668 |
| 16 | 0.5255 | 2.8836 |
| 8 | 0.5257 | 2.7765 |

Split greedy remains the selected path. Local65536 vocabulary width is power-of-two; only32 candidates per chip are gathered. Physical row padding forB8/16 and the force-argmax controls is included in these numbers. Persistent token output aliases the decoder input. No full-vocabulary gather is used by the selected path.

The alternate common Sampling1D contract comparison and adapted repro remain in ../full_model/sampling_choice.md. This stage changes neither sampler implementation nor semantic parameter contract. Fresh all-batch seeded top-k/top-p buffered-vs-streaming acceptance is in generator_state_b*.json.
