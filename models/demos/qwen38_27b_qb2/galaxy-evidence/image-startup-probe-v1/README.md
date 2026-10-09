# Real TTIS startup handoff check, queued October 9 2026 UTC

The old image-runtime check covers actual imports and the entrypoint's `--help`.
It does not execute TTIS setup. A separate read-only probe now runs the installed
wrapper's real `main()` through imports, model registration, logging/cache setup
and command assembly, intercepting only the final vLLM server call. It checks
every declared argument and environment value against the immutable bundled
ModelSpec and confirms that the generated checkpoint symlink resolves to the
read-only weights mount. It does not load weights, expose TT devices or start HTTP.

TTIS source `54a5511fdcaed2dddc215c20f4ea95dad33dbdb6` is pushed on
`anatarajan/qwen38-galaxy-release-20261009`. Twenty-nine local checks passed:
19 existing packaging tests plus 10 argument/environment regressions. The
argument checks reject extra or duplicate flags, wrong mesh groups, changed
precision, missing options and an unexpected operation timeout. These CPU tests
do not claim that the installed wrapper has passed; that result is pending.

The persistent unit `qwen38-image-startup-probe-v1-20261009.service` was observed
active at PID 1383768, invocation `16226e28ecfe424295ae08508c972347`. It waits
for the exact post-head CPU job to become terminal, requires the original image's
runtime import checks to pass, then acquires `/tmp/tt-device.lock`. This keeps
container imports from overlapping queued hardware measurements. Neither of the
existing CPU or hardware controllers was modified.

The probe checks the original native-policy image configuration
`sha256:23f5f92af192e23cde494bcdac1f2efddc5c3b301345af3997ca3f79036c6af1`;
it does not silently substitute the newer head-policy image. Docker receives no
devices or network, read-only root and weights, temporary cache/log storage,
4 GiB RAM, four CPUs and a 180-second execution limit. The controller waits at
most 22 hours inside a 23-hour service. Cleanup targets only its named/labeled
container, before releasing the coordination lock.

The first stage attempt was sandbox-denied before connecting. The approved retry
launched once. `launch.json`, `source-manifest.json`, `controller.py.txt`,
`probe.py.txt`, `status-at-launch.json` and `service-at-launch.txt` preserve the
exact source and invocation. The job survives disconnect, not reboot. Any pass
will remain a packaging check; container model execution, full accuracy and Helm
qualification still require separate evidence.
