# Imported image identity repaired; runtime checks restored

The original native v5 image transfer and Docker load succeeded, but its
controller looked up the OCI config digest. This Docker image store uses the
manifest digest as `Id`; the config-digest lookup failed before runtime checks.
No image corruption or accelerator failure was established.

The new TTIS `scripts/release/verify_local_oci_image.py` checks both pinned
metadata blobs against their content hashes, their manifest-to-config link,
Docker identity, platform, ordered root filesystem layers and runtime config.
It accepts the config or manifest form of immutable Docker image ID and does
not use a mutable tag. Eleven regression tests passed locally and on the host.
The live metadata check passed for all **46 layers**. It reads metadata only;
it does not rehash layer payloads or replace the prior archive-transfer hash.
No container or accelerator was started by this identity check.

TTIS source, tests and instructions are pushed at
`52390d1b0d4015fbde230fbcfb7e357b3b84e24c` on
`anatarajan/qwen38-galaxy-release-20261009`. This is an operator-check update,
not a rebuild of the image. The native v5 image still contains TTIS `e0e05bad...`
and model `0abdc340...` as recorded by its labels and original build evidence.

- Manifest / local Docker ID:
  `sha256:0b11f045bf089088a62b6e3c1aeb9b64cc72b74a632935f25203609b1e023579`
- OCI config:
  `sha256:23f5f92af192e23cde494bcdac1f2efddc5c3b301345af3997ca3f79036c6af1`
- Archive SHA256 previously verified on transfer:
  `755626b044cea12f390fd343439dbfe4af920aba6e7d156a26490c478325d648`

Persistent `qwen38-image-startup-probe-v2-20261009.service` is now waiting on
the exact chunked-state invocation `4c0c8fa5f3b249f79fee72c77d1c137f`, which itself
waits for BFP8 GPQA and its completion auditor. Startup-probe PID was 1468886,
invocation `fea7f95be0e041bab4de9e6a013458e7`. It requires successful terminal
receipts and clean shutdown, then acquires the shared hardware coordination
lock to avoid CPU contention with measurements.

The recovered sequence verifies identity again, runs the image's source/import
checker and entrypoint help, then exercises the standard TTIS runtime-spec,
ModelRegistry, environment, cache and checkpoint setup through its vLLM handoff.
It does not execute model inference. Containers have no accelerator devices or
network, a read-only root, bounded tmpfs, 4 GiB memory and four CPU quota. Only
this job's labeled containers are cleaned up. The job has a 12-hour total bound
and three-minute command bounds; it survives disconnect, not reboot.

These startup checks are **queued, not passed**. Container hardware inference,
accuracy, registry publication and SJC3 installation remain unqualified. The
native v5 image cannot inherit the in-progress BFP8 model's eventual score.
Frozen controller/helper bytes, manifest, launch and service identity are under
`startup/`; the completed metadata validation and native tests are under
`identity/`. Remote roots are `image-identity-v2`, `image-startup-probe-source-v2`
and `image-startup-probe-v2` beneath `/home/ttuser/qwen38-artifacts-20261007`.
