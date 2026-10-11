# Complete-prefix qualification bundle

This is an explicit qualification recipe, not an enabled deployment or release.
Prepare it while the main profiler owns the devices. Only start either generated
launch script when the allocated Galaxy's owner has coordinated device access.
Both phases acquire `/tmp/tt-device.lock`; neither resets devices or clears a
`/tmp/tt-device.dirty` marker. No AgentX, GPQA, installation, weight download or
system configuration change is included.

## Frozen identity

- Model implementation: `d1019c0dc125913a99ba82938d06d5d662e17f7d` on
  `anatarajan/qwen38-prefix-offload-20261010`. Preparation commits may change
  only tests, demo controllers and documentation; `tt/` and `config/` must match.
- Plugin: `13b9777876dc08b268dfc2f627571496484d0f5a` on the separate local branch
  `anatarajan/qwen38-prefix-offload-20261011`. Publication is blocked by upstream
  write permission. Preserved patch: `/private/tmp/qwen38-prefix-plugin-13b9777.patch`.
- Weights: revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, host artifact
  `/home/ttuser/qwen38-artifacts-20261007/checkpoint-pinned-1d4bf0f2`.
- Native runtime: `a08819ddbe23077f8037d3802303939064868ff6` in
  `/home/ttuser/kimi-prefill.Ubx2wY/runtime/qwen38-27b-20261006`.
- Actual physical TP4: `[0, 4, 12, 8]`. Native tests open the existing full
  Galaxy parent mesh before selecting that submesh, so they require exclusive
  access to the Galaxy. HTTP uses one TP4 worker, engine TP1/internal DP1.
- Policy: `config/precision_single_step_compact_gdn_bfp8_all.json`:
  BFP8 weights/KV, BF16 activations/conv, FP32 recurrent state.
- Transfer mode: **batched only**. Old serial passes cannot satisfy these gates.

`bundle.json` hashes every copied source/config/test/controller/plugin file.
Host `seal.json` additionally binds the bundle, generated token-ID prompts,
installed native shared-library bytes and weight index/config/shard size/mtime.
The supplied immutable weight revision is trusted; this does not hash all weight
contents. Each stage revalidates the frozen files. Native continuation receipts
must contain the exact complete model-source and effective-precision map.
Serving startup must log the exact frozen precision policy.

## Prepare without opening devices

From a clean, committed preparation worktree, use its own controller:

```sh
python3 models/demos/qwen38_27b_qb2/demo/prefix_qualification.py freeze \
  --metal /private/tmp/tt-metal-qwen38-prefix-offload \
  --plugin /private/tmp/qwen38-vllm-prefix-offload-20261011 \
  --bundle /private/tmp/qwen38-prefix-qualification-20261011-v1 \
  --task /home/ttuser/kimi-prefill.Ubx2wY/runtime/qwen38-27b-20261006 \
  --weights /home/ttuser/qwen38-artifacts-20261007/checkpoint-pinned-1d4bf0f2 \
  --destination /home/ttuser/qwen38-artifacts-20261007/prefix-qualification-20261011-v1
```

Copy the directory to that exact host destination using `rsync -a`. No archive or
weight copy is needed. Run `seal --bundle DESTINATION` with the existing
`TASK/serving_env/bin/python` and the copied controller. Sealing only reads
native-library/checkpoint metadata and tokenizes generated prompts on CPU.
It writes two explicit scripts but **does not execute them**:

- `start-native.sh`: transfer bytes, 4-layer continuation, full64 continuation.
- `start-http.sh`: owned loopback HTTP server, probes, restart, cleanup test.

Preserve this immutable directory and use a new name for each attempted bundle.
A started phase refuses to overwrite its existing results. Failed runs do not
silently retry or accept another run's receipts.

## Device gates and resource limits

`start-native.sh` creates a persistent user systemd unit with a 2-hour hard
limit. Individual test limits are 15 minutes for packed transfer, 20 minutes for
4 layers, and 60 minutes for full64. These are upper bounds, not runtime estimates.
The lock wait is at most 10 minutes. Full64 checks 4096 prefix tokens, a 32-token
suffix, 32 teacher-forced traced decode steps and an active cached-frontier
restore. It preserves neighbour slots and trace buffer addresses.

`start-http.sh` first verifies that the *same bundle* passed all three native
gates, including clean completion, physical IDs, exact source/precision and
receipt hashes. Its persistent unit has a 2.5-hour hard limit and two startup
limits of 30 minutes each. It listens only on `127.0.0.1:18086` with B16 admission,
32K max context, 4096-token chunks and a task-owned 2 GiB checkpoint quota.
Regular attention-only APC and async scheduling remain disabled; the complete
hybrid connector and experiment flag must be supplied explicitly.

Both units have a 192 GiB memory high watermark, 256 GiB hard memory limit,
2 GiB per-file cap and 8 GiB task-artifact guard. The controller requires at least
12 GiB free disk, stops its own process group, and marks the device dirty after
an acquired-lease failure. It never attempts automatic recovery. Invocation
receipts record only selected runtime settings, not inherited credentials.
Units survive an SSH/client disconnect; they do not promise reboot resumption.
A forced process-group kill is an unsuccessful cleanup, not a passing gate.

## HTTP assertions and interpretation

All completions use identical generated token-ID prompts, greedy host sampling,
64 required output tokens, `return_token_ids=true` and `ignore_eos=true`.
Distinct salts create isolated cold baselines; matching salts select warm reuse.

1. Exact token IDs/text for cold versus warm requests with 1/31/32/33-token
   suffixes after the same 4096-token prefix. Warm admission must increase
   vLLM's external-prefix hit counter by at least 4096 tokens.
2. Concurrent client submissions must match their isolated outputs. Cancel one
   actual streaming request after its first output while its neighbour finishes;
   then issue a follow-up request and compare its output.
3. Corrupt the completion marker of this task's derived checkpoint. The worker
   must fail restore, native vLLM must recompute, and the file must be republished
   byte-for-byte. Deleting a checkpoint must also recompute and republish.
4. Fill the configured cache-accounting quota with a task-owned sparse fixture.
   Capture must decline without failing inference. This does not fill the disk
   or simulate a real ENOSPC failure. The fixture is removed afterward.
5. Stop the owned server, restart with the same immutable identity, and verify
   exact output plus an external-prefix hit from the persisted checkpoint.
6. Stop the second server and rerun the physical transfer gate to check device
   health after serving, without a reset.

HTTP receipts record total nonstream response latency, **not TTFT**. Cold
latency includes capture, so its comparison with warm latency is not a clean
no-cache prefill speedup. The previous serial standalone result was 3.542 s
restore versus 1.185 s prefill; batched timing is still unmeasured. Batching
reduces completion fences but retains Python per-window submissions and
read-before-write restore, so a large speedup is not guaranteed.

Concurrent submissions alone do not prove physical batch overlap, a slot
permutation, or scheduler preemption. Those remain explicit serving gates with
worker evidence before default enablement, along with repeated runs, shared
multi-replica contention and system throughput. Native neighbour-slot checks
and CPU lifecycle tests are complementary evidence. File reads may hit the OS
page cache; this is persisted-file correctness, not measured physical SSD rate.
Multimodal/prompt-embedding/LoRA/prompt-logprobs reuse remains bypassed.
Do not merge or enable serving defaults until the complete lifecycle is
consistently qualified. AgentX waits for integrated prefix and SSD; full GPQA
waits for measured B16/32K/TP4 reaching 25 TSU.
