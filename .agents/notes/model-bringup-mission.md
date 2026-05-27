# Agent-Driven TTNN Model Bringup Mission

The overall mission is to take a HuggingFace model from reference implementation to a TTNN implementation that can run, be checked, be optimized, and eventually serve through the shared vLLM path. Today you are working on one part of that. The skills in this directory are onboarding context for capable, reasoning, introspective and intelligent agents, not scripts to execute mechanically. Read widely and deeply - to really understand the APIs you are using it's often helpful to read their implementation e.g. the actual op implementations as well. tt-metal and ttnn are new APIs so don't assume you know how they will behave without looking.

## Implementation

Work accumulates under one per-model directory:

```text
models/autoports/<model>/
  tt/
  tests/
  doc/
```

The main stages are:

- `functional-decoder`: implement a correct TTNN decoder layer path for each meaningful layer kind.
- `optimize-decoder`: preserve the functional contract while improving latency, layout, precision, and data movement.
- `multichip-decoder`: parallelize the decoder across the target hardware without weakening correctness.
- `productize`: wrap the decoder into a full model, generator, vLLM adapter, and readiness-check path.

Later stages build on earlier ones.

Read `.agents/notes/output-files.md` to understand the output files required.

## Create a Clean De-Novo Implementation

Although you should learn from other code, do not import or blindly copy it. What we want here is a de-novo implementation inspired by the best that has come before, but it should stand alone as a complete ttnn implementation. Helper code and utilities may be imported and re-used as you see fit, but we'd like to keep the model forward pass implemented here directly, cleanly and readably in one place. Make it readable, simple and elegant. Each line of code should be able to justify its inclusion; brevity can be beauty as long as it is not obfuscating something important.

Always use optimized/fused ttnn operations when they fit, e.g. ttnn provides paged sdpa operations for prefill and decode - have a look to see which ones other models use and what ttnn provides especially in ttnn.transformer. When they fit you should generally use these instead of hand-implementing one based on primitives.

## Working Style

Be curious and tenacious. Read around the codebase, inspect the HF source, compare against nearby TTNN models, and debug from evidence. Skills may point to useful files and norms, but you, the model and the hardware decide the implementation. Be your best self and at all times exercise your best judgement in pursuing our goals. You own the delivery of the output. Rather than trying a sequence of steps mechanically and reporting failure if they don't work, be curious about failures or inconsistencies, follow them down to the root cause. Fix and work around them and above all have a good time doing work you are proud of!

## Brief Notes on Developing with TTNN

Use `tt-smi -r` to reset the device before you begin. You have exclusive access to this device and server and have explicit permission to perform long-duration runs so don't limit yourself to smoke tests when real tests are called for. TT devices are exclusive so be sure never to reset whilst the device is in use and never to try to run multiple things on device at the same time. If a device hangs you can run tt-triage to find out where and why. After a hang, a crash or killing a run you should reset the device with `tt-smi -r` or subsequent runs may fail.
