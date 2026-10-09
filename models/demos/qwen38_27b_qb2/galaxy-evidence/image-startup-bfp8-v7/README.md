# BFP8 image import and startup queue

The completed [v7 image](../image-build-v7/README.md) has a persistent follower
on 10.228.203.98. At 10:40 UTC it was alive and waiting; no archive download,
image import, container startup or accelerator operation had started.

- Unit: `qwen38-image-startup-bfp8-v7-20261009.service`.
- PID at launch: 1582084; invocation `c3068a78ac924fd3a1de1c847be0e53c`.
- Predecessor: `qwen38-precision-perf-v1-20261009.service`, exact invocation
  `122b1077bed2438d9802360ab6277514`, completed receipt and hardware cleanup required.
- Result root: `/home/ttuser/qwen38-artifacts-20261007/image-startup-bfp8-v7`.
- Frozen source: adjacent `image-startup-bfp8-source-v7`; the three retained
  Python sources are gzipped here, with original SHA256 values in the manifest.
- TTIS helper source: `bbb1ca07e7cd0f97af47745a1686545a16089b9c`.

After its predecessor finishes, the job acquires `/tmp/tt-device.lock`, checks
host-disk headroom, copies only the pinned archive over certificate-verified
TLS, and verifies exact byte count and full SHA256. It fsyncs and atomically
renames the archive before importing. The disk check budgets the archive,
compressed image store, 21 GiB uncompressed storage and an 8-GiB reserve; it
does not delete other files to make room. Its correctness does not depend on
Docker deduplicating existing layers.

It then checks the immutable imported manifest/config using the previously
tested OCI verifier, runs source/import verification and entrypoint help, and
executes the real TTIS setup until its final vLLM handoff. All containers have
no network or TT devices, read-only roots, read-only mounted weights, bounded
task-owned tmpfs, four CPUs and 4 GiB RAM. Startup checks validate arguments,
environment and checkpoint path; they do not load weights or start inference.
The controller records `accuracy_qualified=false` even when these checks pass.

The transfer server on .34 serves only this archive at its digest path to .98,
with a fresh pinned certificate and an 18-hour lifetime. The controller has a
14-hour predecessor/lock wait, a 16-hour service bound and explicit bounds on
copy, import and checks. Both survive disconnect, not reboot. It shares the
lock with the earlier native-image startup job rather than assuming sibling
followers run in a particular order.

Validation before launch: Python syntax checks for controller/server/staging,
remote helper CLI import, all frozen source hashes, and a successful TLS HEAD
request with the exact archive length. The reused helpers have 44 packaging
tests including 11 OCI identity cases; no new container result is claimed.
The TLS public certificate is retained for reproduction; no private key is
published. The private key remains only in the task-owned transfer directory.

The first stage attempt stopped before creating any transfer files because
systemd had collected the successful build unit and cleared its InvocationID.
Inspection proved the unit inactive, its completed build receipt present and
its owned builder stopped. The staging guard now accepts that terminal case,
while rejecting a different nonempty invocation, running service or missing
completed artifact. Existing failed image attempts were not overwritten.
