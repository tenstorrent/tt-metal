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

No completed image digest, registry publication, head GPQA pass, or container
hardware qualification is claimed. The build survives disconnect; only a
successfully preserved final archive survives reboot.
