# BFP8 experimental image build

The full BFP8 decoder [passed eight-replica G0](../decoder-g0-v1/README.md).
Its exact-SHA image finished building on host 10.228.203.34 at 10:33:37 UTC,
after 14m58s including archive preservation. The build's source/import checks
passed in both builder and final runtime stages. Native GPQA continues on
10.228.203.98. This image is **not accuracy-qualified** and has not run inference.

- OCI manifest: `sha256:79f7b4469a6ec2bcce5204399b37b2aeced8be7f260dd98f7007ad41f8813055`.
- OCI config: `sha256:2f70ab2ae93b7edb362340ef3c241f1577a49d6795548c7d62cdde4f4e5b1ecd`.
- Archive: 6,011,314,688 bytes, SHA256
  `603a03b83772c8f47f7817d838316657ef6b4a77e972d9f9580559d4db2550dd`.
- Durable artifact: `/home/ttuser/qwen38-release-build-20261009-v7/artifact/qwen38-image.oci.tar`.

- Model source: `20619e008a236aaf393937b222a60a5b03e49cdc`, branch
  `anatarajan/qwen38-bfp8-control-runtime-20261009`.
- TTIS: `bbb1ca07e7cd0f97af47745a1686545a16089b9c`, branch
  `anatarajan/qwen38-galaxy-release-20261009`.
- Metal: `a08819ddbe23077f8037d3802303939064868ff6`.
- Plugin: `b7e4292e4193cba20abe9c7c68ce489201b2e36b`.
- Checkpoint: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, mounted separately.
- Precision: `precision_accurate_decode_bfp8_all.json`, fingerprint
  `d12c863525c83159ff13d43170178987679e0c82a5e83799ad5cf3566ec836ea`.

Bundle preparation verified the exact G0 hash against committed model bytes.
The TTIS wrapper, startup and image-identity tests passed 44 cases; runtime and
authenticated Helm checks cover this BFP8 policy explicitly. These are packaging
checks, not container inference. The complete G0 file is retained in the linked
evidence directory rather than duplicated here; its hash is in the manifest.
The frozen image verifier is stored as `verify_image.py.gz` to preserve its
original bytes through repository formatters; decompress it when reconstructing
the build bundle. The executed build context contains the uncompressed file.

Unit `qwen38-release-build-v7-20261009.service`, PID 3977785, invocation
`ea312e233d9e4dd5995f81ccadd7f564`, ran the same bounded image-only builder as
the prior successful v6 build. No accelerator devices are exposed. Limits:
192 GiB host RAM, 24 CPUs, four hours. The read-only preflight found 22.04 GB
host-disk free and 291.91 GB tmpfs free; the controller independently checks
its disk/tmpfs guards. It preserved the OCI archive on host disk with matching
source/destination checksums, fsync and atomic rename, then stopped its own
builder. Systemd reports successful completion and the owned builder is
stopped. The preserved artifact survives reboot.

Remote root: `/home/ttuser/qwen38-release-build-20261009-v7`. The launch,
context manifest, frozen controller, initial service state, exact bundle and
observed test result are retained here. No native installation, current serving
process, registry or SJC3 deployment has been changed by this build.

The [persistent import/startup follower](../image-startup-bfp8-v7/README.md)
waits for the current accuracy/state/performance chain before copying or
importing this image. Runtime checks, container hardware qualification,
registry publication and SJC3 Helm deployment remain separate gates.
