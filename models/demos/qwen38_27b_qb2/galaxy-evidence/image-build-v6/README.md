# Higher-precision-head experimental image build

The separate persistent v6 build started on the idle .34 host on October 9,
2026, after the head policy passed eight-replica G0. It uses the existing
no-device BuildKit recipe, bounded to four hours, 24 CPUs and 192 GiB RAM.
Its 170-GiB build workspace lives in removable tmpfs; no NFS is used.

Inputs:

- TTIS: `2485b039be071f75fa29adff6c84ecc87d60359e`.
- Metal runtime: `a08819ddbe23077f8037d3802303939064868ff6`.
- Model subtree: `d3e8d6021f7aadcb28ba3903d79bd04b288a2819`.
- Plugin: `b7e4292e4193cba20abe9c7c68ce489201b2e36b`.
- Precision: `precision_accurate_decode_bfp8_head.json`.
- Actual G0 receipt SHA-256:
  `02517a16c9964c37ca42f9b69d0ec34f89e3e19f93f76fe76c2045ce406f9fbb`.

Unit `qwen38-release-build-v6-20261009.service` was observed active with PID
3806410 and invocation `95040bf8652248bca40bb52f13caeb76`. The environment
probe passed and the base image was being unpacked. The source manifest covers
41 exact context files. `state.json` is an initial snapshot, not a build result.

On successful build, the controller copies the OCI archive from RAM to
`/home/ttuser/qwen38-release-build-20261009-v6/artifact/qwen38-image.oci.tar`,
checks both complete SHA-256 digests, fsyncs the file and directory, and only
then records a durable artifact. It retains an 8-GiB disk reserve. The original
native-policy archive and its queued runtime checks remain unchanged.

The first staging attempt was sandbox-denied before connecting. A subsequent
receipt-newline edit produced a remote Python syntax error before any remote
mutation; the payload was corrected and compile-checked before the successful
launch. Only one v6 build service was launched.

## Completion, 08:14:41 UTC

The build completed in 15m25s, passed source/import checks in both image stages,
and preserved the 6,010,795,008-byte archive on host disk. A later independent
read rechecked the complete archive checksum and the embedded manifest/config
digests and source labels. The service is inactive with PID 0; this is a completed
build, not a lost connection.

- OCI manifest: `sha256:c8ed7a5a17b4b400c84bbb82a52bd125a56b4daf5fbb150b699af62ffee169b1`.
- Image configuration: `sha256:87e5c8cf71db455e1d19082a589663d18113ee3c5805d7e0cf346ac33286bb18`.
- Archive SHA-256: `dd3bd0e91716870d13bd0ad2d67baf35664272cf946132adec77760ccf71ec25`.

[Final state](final-state.json), [build metadata](build-metadata.json),
[independent archive audit](oci-audit.json), and `build.log.gz` retain the proof.
The earlier `state.json` is deliberately retained as a launch snapshot. About
20.5 GiB remained on the build host after preservation; no unrelated images or
data were removed. The durable archive survives reboot.

This remains unqualified: full head GPQA had 166 correct out of 194 completed
at 08:28 UTC, so even four additional correct answers cannot meet 177/198.
Registry publication and container hardware qualification have not occurred.
