# Experimental Qwen image built and preserved

Build completed at 05:09:16 UTC, Oct 9, 2026, on `10.228.203.34`.
`qwen38-release-build-v5-20261009.service` exited successfully. Native Metal
compilation, dependency installation, source/precision/version verification
and imports passed in both image stages. No accelerators were opened.

- OCI manifest: `sha256:0b11f045bf089088a62b6e3c1aeb9b64cc72b74a632935f25203609b1e023579`.
- Archive size: 6,010,501,632 bytes.
- Archive checksum: `755626b044cea12f390fd343439dbfe4af920aba6e7d156a26490c478325d648`.
- Durable host path: `/home/ttuser/qwen38-release-build-20261009-v5/artifact/qwen38-image.oci.tar`.

The archive checksum identifies the tar file; it is not the OCI manifest digest.
`qwen38-preserve-image-v5-20261009.service` copied the RAM-backed build output
to task-owned host disk, fsynced it, compared both complete SHA-256 checksums
and exited successfully. The artifact survives reboot. The remaining RAM copy
and stopped builder container are task-owned and removable; they were retained.

TTIS source is `e0e05bad5361d7c170068b3ad7b4df27de192250`, Metal runtime
`a08819ddbe23077f8037d3802303939064868ff6`, model subtree
`0abdc3403f039c46becef335ad02db99237593f8`, plugin
`b7e4292e4193cba20abe9c7c68ce489201b2e36b`. The effective policy is the
native BFP4/LoFi-head control, whose full GPQA score is 170/198. The queued
higher-precision head experiment does not change this image.

This is **built, unqualified, and not yet published to a registry**. A local
manifest digest does not make it pullable by Helm. Registry publication,
container hardware/API/evaluation checks and SJC3 Helm validation remain.
Do not promote the image based on successful build/import checks.

`build.log.gz` preserves execution evidence; JSON files record immutable
build inputs, the actual output digest and durable-copy verification.

The published preservation JSON adds only the missing final newline.
