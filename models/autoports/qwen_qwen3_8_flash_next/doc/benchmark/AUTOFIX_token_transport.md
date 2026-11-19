# Exact-token benchmark transport repair

Diagnosis: AUTODEBUG_token_transport.md. The earliest inconsistency is transporting client-sized random tokens through text and asking a differently resolved server tokenizer to segment them again.

Added model-local tools/benchmark_client.py. It invokes the unmodified installed upstream `vllm bench serve` entrypoint, intercepts only RandomDataset.sample after upstream generation, encodes each generated text with that same client tokenizer, checks length equals the upstream declaration, checks distinct prompts, and replaces its prompt with those token IDs. The existing upstream completion HTTP function supports this representation. Generation, request scheduling, server-usage accounting, performance metrics and output artifacts stay upstream. The wrapper also retains every input token ID in `<result-filename>.input-token-ids.json` before timing begins. Upstream server-alignment helper skips token-ID prompts because re-tokenization is unnecessary. Any local roundtrip mismatch fails explicitly.

Set benchmark_command to:

```json
["/home/ttuser/qwen38-flash/vllm-benchmark-env/bin/python", "models/autoports/qwen_qwen3_8_flash_next/tools/benchmark_client.py", "bench", "serve", "--tokenizer", "/home/ttuser/qwen38-flash/hf/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/812d96e10a0dd82b69d749e68693698cf6c36daa"]
```

CPU checks validate two distinct prompts each at 128, 4096 and 32768 input tokens, retaining 128 outputs and exact upstream token sequences. Evidence is setup/random-token-transport-cpu.json. Parent must validate server-reported counts on the next owned setup pilot before the timed run. No hardware result or performance claim is made here. Installed client, plugin, working environment and server/model source remain unchanged.
