# Prefix serving preparation: sealed, not launched

Host: `ttuser@10.228.203.98`. Frozen task directory:
`/home/ttuser/qwen38-artifacts-20261007/prefix-qualification-20261011-v1`.
Preparation source commit: `0abc84299014e20deb366102648fb6bdd77a421e`.
The source/config bytes are exactly model `d1019c0d` plus plugin `13b9777`.

Copied from that host after successful **CPU-only** sealing:

- `bundle.json`: 283 source/test/config/plugin file hashes; SHA256
  `30c67c050b982e7e9751d44f2b67cc3e860f519391fdf05c729ce1312e406e59`.
- `seal.json`: local weight metadata, installed native-library hashes, prompt
  hash and bundle binding; SHA256
  `c4600a16336d813579e94513c1f95348a6d7a531901541027edfac0e1dfe0c94`.
- `start-native.sh`, `start-http.sh`: generated explicit persistent recipes;
  neither was executed. These copies contain the actual host paths.

All **57 prefix CPU tests passed** on the host's disposable interpreter in
0.782 s, including real process-group cleanup on Linux. The executed command,
from the frozen `source` directory, was:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/ttuser/qwen38-artifacts-20261007/plugin-multimodal-env-v1/bin/python \
  -m unittest discover -s models/demos/qwen38_27b_qb2/tests/unit \
  -p 'test_prefix*.py' -q
```

This result was collected from the command's returned stdout, not a native
hardware receipt. Local validation passed 56 tests with the Linux-only test
skipped. Explicit pre-commit checks passed before committing preparation.
No device, reset, service launch, package installation or existing deployment
mutation occurred. Parent profiling remained active under the device lock.

Only start the native script after coordinating access with that lock's owner.
The HTTP script independently rejects absent, changed, serial or wrong-source
native results. See the [full recipe](../../experiments/PREFIX-QUALIFICATION-BUNDLE.md)
for bounds and remaining proof limits. A sealed preparation bundle is not a
serving pass, speedup result, release/default enablement or AgentX authorization.
