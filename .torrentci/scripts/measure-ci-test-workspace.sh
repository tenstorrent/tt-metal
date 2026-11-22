#!/bin/sh
# Temporary (torrentci#3506): measure the workspace a cold ci-test chain-image
# build needs. Replays the chain-image runner's layout: buildkitd root at
# /workspace/.buildkit, one gzip OCI layout tar per bake target under
# /workspace/.torrentci-chain-image, every target cold. Contexts are extracted
# outside the measured directories, as the runner extracts them into its own
# contexts-store volume. du samples every 10 s.
set -eu
ws="$(pwd)"
root="$ws/.buildkit"
layouts="$ws/.torrentci-chain-image"
ctx="/tmp/contexts"
mkdir -p "$root/run" "$layouts" "$ctx"
wrapper=/tmp/runc-drop-net-raw
cat >"$wrapper" <<'W'
#!/bin/sh
set -eu
runc="$(command -v buildkit-runc || command -v runc)"
drop() { sed -i -e 's/"CAP_NET_RAW",//g' -e 's/,"CAP_NET_RAW"//g' -e 's/"CAP_NET_RAW"//g' "$1/config.json"; }
prev=""
for a in "$@"; do
  case "$prev" in --bundle | -b) drop "$a" ;; esac
  case "$a" in --bundle=*) drop "${a#--bundle=}" ;; esac
  prev="$a"
done
exec "$runc" "$@"
W
chmod 0755 "$wrapper"
addr="unix://$root/run/buildkitd.sock"
buildkitd --root "$root" --addr "$addr" --oci-worker-binary "$wrapper" >/tmp/buildkitd.log 2>&1 &
bk=$!
peak=/tmp/peak
echo "0 0 0" >"$peak"
(
  while :; do
    r="$(du -sxk "$root" 2>/dev/null | cut -f1)"
    l="$(du -sxk "$layouts" 2>/dev/null | cut -f1)"
    t=$((r + l))
    # shellcheck disable=SC2046
    set -- $(cat "$peak")
    if [ "$t" -gt "$1" ]; then
      echo "$t $r $l" >"$peak"
      echo "measure: new peak $((t / 1024)) MiB (buildkit root $((r / 1024)) MiB, layouts $((l / 1024)) MiB)"
    fi
    sleep 10
  done
) &
sampler=$!
trap 'kill $sampler $bk 2>/dev/null || true' EXIT
i=0
until buildctl --addr "$addr" debug workers >/dev/null 2>&1; do
  i=$((i + 1)); [ "$i" -lt 60 ] || { cat /tmp/buildkitd.log; exit 1; }; sleep 1
done
args="--opt build-arg:UBUNTU_VERSION=24.04 --opt build-arg:PYTHON_VERSION=3.12 --opt build-arg:TT_SMI_VERSION=6.7.0"
args="$args --opt build-arg:UV_IMAGE=ghcr.io/astral-sh/uv@sha256:9f130421e930bfbdc563c00740466603f8723158e90297d07957c2294640262b"
build() { # name file target [extra args]
  name="$1"; file="$2"; target="$3"; shift 3
  start="$(date +%s)"
  # shellcheck disable=SC2086
  buildctl --addr "$addr" build --progress=plain --frontend dockerfile.v0 \
    --local context=. --local dockerfile=dockerfile --opt "filename=$file" --opt platform=linux/amd64 \
    --opt "target=$target" $args "$@" --metadata-file "/tmp/$name.meta.json" \
    --output "type=oci,dest=$layouts/$name.oci,compression=gzip,oci-mediatypes=true" >"/tmp/$name.log" 2>&1 \
    || { tail -n 200 "/tmp/$name.log"; exit 1; }
  digest="$(sed -n 's/.*"containerimage.digest": *"\(sha256:[0-9a-f]*\)".*/\1/p' "/tmp/$name.meta.json")"
  mkdir -p "$ctx/$name" && tar -xf "$layouts/$name.oci" -C "$ctx/$name"
  echo "$digest" >"$ctx/$name.digest"
  echo "measure: built $name in $(($(date +%s) - start)) s: layout $(($(du -k "$layouts/$name.oci" | cut -f1) / 1024)) MiB, root $(($(du -sxk "$root" | cut -f1) / 1024)) MiB"
}
tools="ccache mold doxygen clangbuildanalyzer iwyu gdb cmake yq zstd curl sfpi openmpi"
for t in $tools; do build "$t" Dockerfile.tools "$t"; done
build ci-test-venv Dockerfile.python ci-test-venv
named=""
for t in $tools ci-test-venv; do
  key="$t-layer"
  named="$named --oci-layout $t=$ctx/$t --opt context:$key=oci-layout://$t@$(cat "$ctx/$t.digest")"
done
# shellcheck disable=SC2086
build ci-test Dockerfile ci-test $named
sleep 11
# shellcheck disable=SC2046
set -- $(cat "$peak")
echo "measure: PEAK workspace $(($1 / 1024)) MiB (buildkit root $(($2 / 1024)) MiB, layouts $(($3 / 1024)) MiB)"
echo "measure: final buildkit root $(($(du -sxk "$root" | cut -f1) / 1024)) MiB, layouts $(($(du -sxk "$layouts" | cut -f1) / 1024)) MiB, contexts $(($(du -sxk "$ctx" | cut -f1) / 1024)) MiB"
du -sxk "$layouts"/* | sort -n
buildctl --addr "$addr" du | tail -n 3
