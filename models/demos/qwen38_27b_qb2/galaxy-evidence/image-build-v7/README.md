# BFP8 experimental image build

The full BFP8 decoder [passed eight-replica G0](../decoder-g0-v1/README.md).
Its exact-SHA bundle is prepared and its image is building on CPU-only host
10.228.203.34 while serving startup/GPQA runs on 10.228.203.98. This is **not
accuracy-qualified**; no new image digest is available until the build finishes.

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
`ea312e233d9e4dd5995f81ccadd7f564`, runs the same bounded image-only builder as
the prior successful v6 build. No accelerator devices are exposed. Limits:
192 GiB host RAM, 24 CPUs, four hours. The read-only preflight found 22.04 GB
host-disk free and 291.91 GB tmpfs free; the controller independently checks
its disk/tmpfs guards. It will preserve the OCI archive on host disk with a
checksum and atomic rename, then stop its own builder. It survives disconnect,
not reboot; the preserved artifact will survive reboot.

Remote root: `/home/ttuser/qwen38-release-build-20261009-v7`. The launch,
context manifest, frozen controller, initial service state, exact bundle and
observed test result are retained here. No native installation, current serving
process, registry or SJC3 deployment has been changed by this build.
