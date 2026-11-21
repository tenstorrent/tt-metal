# Llama-3.1-8B on Blackhole P100

Llama-3.1-8B for a single Blackhole P100. The model is TT's `models/ttt_compat` Llama with the layers that run on dedicated
`ttnn.experimental.llama31_8b` ops swapped in through subclasses; everything else (weights, tokenizer, generator, demo flow)
is imported unchanged.

```
pytest models/demos/llama31_8b_p100/demo/demo.py -k "performance-ci-token-matching"
```

Only Llama-3.1-8B on one P100 (7 DRAM banks) takes these ops (`tt/p100.py`); any other card, mesh or model runs the
`models/ttt_compat` path unchanged. Each op's wiring is one module, `tt/ops_<op>.py`, reached only through `tt/wiring.py`:
without that module (or without the op built) the layer makes the original TT call in its place.

```
pytest models/demos/llama31_8b_p100/tests/test_wiring.py
```
