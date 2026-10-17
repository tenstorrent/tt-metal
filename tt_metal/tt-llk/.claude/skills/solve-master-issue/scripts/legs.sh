#!/bin/bash
# usage: legs.sh request --run-dir D --arch wormhole|blackhole --label <short-name> (--branch <name> --sha <sha> | --patch <file>)
#                        --cmd '<shell command>' [--cwd <dir in the repo, default tt_metal/tt-llk/tests>] [--timeout <s, 1800>] [--env K=V ...]
#        legs.sh serve   --run-dir D --arch wormhole|blackhole --repo <tt-metal checkout on this box>
#                        [--worktree-root <dir, default parent of --repo>] [--once] [--interval <s, 30>]
#        legs.sh wait    --run-dir D <request-id> [--timeout <s, 7200>]
#        legs.sh status  --run-dir D [--interval <s, 30>]
# Test legs across boxes. A worker that owns a fix often needs a fail-without/pass-with run on the
# architecture it does not have; instead of CI it asks the worker on the other box. There is no network
# protocol: everything is a file under D/legs/, where D is the run dir on the shared filesystem
# ($MASTER_ISSUE_DIR/<run-name>, today /proj_sw/user_dev/amahmud/master-issues/audit-llk-whbh).
#   D/legs/<ts>-<label>-<rand>/   one request; the dir name is the request id (sortable: oldest first)
#     request.json                written last and atomically — a dir without it is not a request yet
#     patch.diff                  patch mode only (copy of --patch)
#     claimed.d/  claimed         a server took it: mkdir claimed.d is the atomic claim, `claimed` says host+pid
#     leg.log                     stdout+stderr of the command (plus any refusal message)
#     result.json                 written atomically when the leg is done: exit, head, log, revert_ok
#   D/legs/serve-<arch>.log       one line per server event
#   D/legs/serve-<arch>.alive     heartbeat, rewritten on every scan and every --interval during a leg:
#                                 `<hostname> pid=<pid> <UTC time> busy=<request-id or ->`
#   D/STOP or D/legs/STOP-<arch>  asks the serve loop to exit after the current leg
# The requester first pushes its branch to origin (branch mode: the server fetches origin/<branch> and
# checks out <sha> in a disposable worktree <worktree-root>/wt-legs-<arch>, created from origin/main on
# first use, with tests/sfpi and tests/.venv symlinked from --repo), or produces `git diff > x.diff`
# (patch mode, for ttnn tests that need a built tree: the server applies it to --repo itself, runs, and
# reverts it, holding the device lock from before the apply to after the revert; refused when --repo is
# dirty beforehand — it never cleans a tree). Then `legs.sh wait <id>`
# blocks until result.json exists and exits with the leg's exit code. The server never pushes, never
# touches GitHub, and never runs anything but the requested command. That command runs with cwd
# <tree>/<cwd>, CHIP_ARCH=<arch> and the request's env, under the box's device lock
# ${TT_DEVICE_LOCK:-/tmp/tt-device-<host>.lock}, after a `tt-smi -r 0` in the same lock hold (the reset
# is skipped when TT_DEVICE_LOCK is set — that is the test harness). A request whose server died stays
# "claimed" with no result: `rm -r claimed.d claimed` re-opens it. `legs.sh status` prints one line per
# arch that has a heartbeat (`serve <arch>: <host> pid=<pid> last seen <N>s ago (busy=<id|->)`, marked
# STALE when older than 3 x --interval, default 90 s) and then one line per request.
# One serve loop per arch per box is expected; start it detached and leave it running:
#   nohup setsid $SKILL/scripts/legs.sh serve --run-dir <D> --arch blackhole --repo /localdev/<user>/tt-metal > /dev/null 2>&1 &
# Exit codes: request 0 (prints the id) · 2 bad args.   serve --once 0 served one · 3 nothing open · 2 bad args.
#   wait exits with the leg's code: 0 pass · N fail · 124 command timed out (137 if it ignored TERM) ·
#   96 server refused: --repo dirty before applying the patch · 97 tree not prepared: sha not reachable,
#   patch does not apply, cwd missing · 125 wait itself timed out with no result.   status 0.
set -uo pipefail
die() { echo "legs.sh: $*" >&2; exit 2; }
now() { date -u +%FT%TZ; }
log() { echo "$(now) $*" >> "$slog"; }
beat() { echo "$(hostname) pid=$$ $(now) busy=${1:--}" > "$legs/serve-$arch.alive"; }
json_write() {  # json_write <path> k=<string> k:=<json> env+K=V ...   (temp file in the same dir, then rename)
    python3 -I - "$@" <<'PY'
import json, os, sys, tempfile
path, fields, env = sys.argv[1], {}, {}
for a in sys.argv[2:]:
    if a.startswith('env+'):
        k, v = a[4:].split('=', 1); env[k] = v; continue
    k, v = a.split('=', 1)
    if k.endswith(':'): fields[k[:-1]] = json.loads(v)
    else: fields[k] = v
if env: fields['env'] = {**fields.get('env', {}), **env}
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or '.', prefix='.' + os.path.basename(path) + '.')
with os.fdopen(fd, 'w') as f: json.dump(fields, f, indent=1); f.write('\n')
os.chmod(tmp, 0o644); os.replace(tmp, path)
PY
}
load_request() {  # load_request <req dir>: sets r_id r_arch r_label r_requester r_mode r_branch r_sha r_cmd r_cwd r_timeout r_env_sh
    eval "$(python3 -I - "$1/request.json" <<'PY'
import json, shlex, sys
r = json.load(open(sys.argv[1]))
for k in ('id', 'arch', 'label', 'requester', 'mode', 'branch', 'sha', 'cmd', 'cwd'):
    print(f"r_{k}={shlex.quote(str(r.get(k) or ''))}")
print(f"r_timeout={int(r.get('timeout') or 1800)}")
print('r_env_sh=' + shlex.quote(''.join(f'export {k}={shlex.quote(str(v))}\n' for k, v in (r.get('env') or {}).items())))
PY
)"
}

cmd_request() {
    local D= arch= label= branch= sha= patch= cmd= cwd=tt_metal/tt-llk/tests timeout=1800 envs=() mode e
    while [ $# -gt 0 ]; do
        [ $# -ge 2 ] || die "$1 needs a value"
        case $1 in
        --run-dir) D=$2;; --arch) arch=$2;; --label) label=$2;; --branch) branch=$2;; --sha) sha=$2;;
        --patch) patch=$2;; --cmd) cmd=$2;; --cwd) cwd=$2;; --timeout) timeout=$2;; --env) envs+=("env+$2");;
        *) die "unknown arg $1";; esac; shift 2
    done
    [ -n "$D" ] && [ -n "$label" ] && [ -n "$cmd" ] || die "--run-dir, --label and --cmd are required"
    case $arch in wormhole|blackhole) ;; *) die "--arch must be wormhole|blackhole";; esac
    [[ $label =~ ^[A-Za-z0-9._-]+$ ]] || die "--label: letters, digits, . _ - only"
    [[ $timeout =~ ^[0-9]+$ ]] || die "--timeout must be an integer (seconds)"
    for e in "${envs[@]}"; do [[ ${e#env+} =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || die "--env wants K=V, got '${e#env+}'"; done
    if [ -n "$patch" ]; then
        [ -z "$branch$sha" ] || die "--patch excludes --branch/--sha"
        [ -s "$patch" ] || die "patch file $patch is missing or empty"
        mode=patch
    else
        [ -n "$branch" ] && [ -n "$sha" ] || die "need --branch and --sha (pushed to origin), or --patch"
        mode=branch
    fi
    mkdir -p "$D/legs" || die "cannot create $D/legs"
    D=$(readlink -f "$D")
    local id="$(date -u +%Y%m%dT%H%M%SZ)-$label-$(head -c 2 /dev/urandom | od -An -tx1 | tr -d ' \n')" pfile=
    local req=$D/legs/$id
    mkdir "$req" || die "cannot create $req"
    [ $mode = patch ] && { cp "$patch" "$req/patch.diff" || die "cannot copy $patch"; pfile=patch.diff; }
    json_write "$req/request.json" id="$id" arch="$arch" label="$label" requester="$(hostname)" created="$(now)" \
        mode="$mode" branch="$branch" sha="$sha" patch="$pfile" cmd="$cmd" cwd="$cwd" "timeout:=$timeout" \
        'env:={}' "${envs[@]}" || die "cannot write $req/request.json"
    echo "$id"
}

claim_next() {  # sets req= the oldest open request for this arch, claimed by us; empty when none
    local f d; req=
    for f in "$legs"/*/request.json; do
        [ -e "$f" ] || break
        d=${f%/request.json}
        { [ -e "$d/result.json" ] || [ -e "$d/claimed.d" ]; } && continue
        load_request "$d" 2>/dev/null || continue
        [ "$r_arch" = "$arch" ] || continue
        mkdir "$d/claimed.d" 2>/dev/null || continue
        echo "$(hostname) $$" > "$d/claimed"
        req=$d; return 0
    done
}
finish() {  # finish <exit> [message]: revert the patch if applied, write result.json, log   (uses serve_one's locals)
    local rc=$1 msg=${2:-} extra=()
    [ -n "$msg" ] && { echo "legs.sh: $msg" >> "$d/leg.log"; extra+=(message="$msg"); }
    if [ "$r_mode" = patch ]; then
        [ -n "$applied" ] && { git -C "$repo" apply -R "$applied" 2>>"$slog"; applied=; }
        if [ -z "$(git -C "$repo" status --porcelain)" ]; then extra+=('revert_ok:=true')
        else extra+=('revert_ok:=false'); log "$r_id WARNING $repo is dirty after the leg — clean it by hand"; fi
    fi
    json_write "$d/result.json" id="$r_id" server="$(hostname)" arch="$arch" mode="$r_mode" tree="$tree" head="$head" \
        started="$started" finished="$(now)" "exit:=$rc" log="$d/leg.log" "${extra[@]}"
    exec 9>&- 2>/dev/null   # release the device lock held across a patch-mode leg (no-op otherwise)
    log "$r_id done exit=$rc${msg:+ ($msg)}"
    beat
}
serve_one() {  # serve_one <req dir>: prepare the tree, run the command under the device lock, write result.json
    local d=$1 tree= head= started l rc beater have_lock=0
    lock=${TT_DEVICE_LOCK:-/tmp/tt-device-$(hostname).lock}
    started=$(now); beat "$r_id"
    log "$r_id claimed label=$r_label mode=$r_mode requester=$r_requester"
    if [ "$r_mode" = branch ]; then
        tree=$wroot/wt-legs-$arch
        if [ ! -d "$tree" ]; then
            git -C "$repo" worktree add -q --detach "$tree" origin/main 2>>"$slog" || { finish 97 "cannot create worktree $tree"; return; }
        fi
        git -C "$repo" fetch -q origin "$r_branch" 2>>"$slog" || log "$r_id fetch origin/$r_branch failed, trying the sha anyway"
        # -f: the serving worktree is disposable; a previous leg's edits to tracked files must not block this one
        git -C "$tree" checkout -q -f --detach "$r_sha" 2>>"$slog" || { finish 97 "sha $r_sha not reachable (is $r_branch pushed to origin?)"; return; }
        for l in sfpi .venv; do
            [ -e "$repo/tt_metal/tt-llk/tests/$l" ] && [ ! -e "$tree/tt_metal/tt-llk/tests/$l" ] &&
                ln -s "$(readlink -f "$repo/tt_metal/tt-llk/tests/$l")" "$tree/tt_metal/tt-llk/tests/$l" 2>/dev/null
        done
        head=$(git -C "$tree" rev-parse HEAD)
    else
        tree=$repo
        # the built tree is shared with the local worker's own apply/revert legs: hold the device lock from
        # before the apply until after the revert, so no one sees (or clobbers) a half-patched tree
        exec 9>"$lock"; flock 9 || { finish 97 "cannot take the device lock $lock"; return; }; have_lock=1
        [ -z "$(git -C "$repo" status --porcelain)" ] || { finish 96 "$repo is dirty, refusing to apply a patch"; return; }
        git -C "$repo" apply --check "$d/patch.diff" 2>>"$slog" || { finish 97 "patch does not apply to $repo @ $(git -C "$repo" rev-parse --short=11 HEAD)"; return; }
        git -C "$repo" apply "$d/patch.diff" 2>>"$slog" || { finish 97 "git apply failed in $repo"; return; }
        applied=$d/patch.diff
        head="$(git -C "$repo" rev-parse HEAD)+patched"
    fi
    [ -d "$tree/$r_cwd" ] || { finish 97 "cwd $r_cwd does not exist in $tree"; return; }
    log "$r_id running in $tree/$r_cwd @ $head timeout=${r_timeout}s"
    ( while :; do sleep "$interval"; beat "$r_id"; done ) & beater=$!   # keep the heartbeat fresh during a long leg
    (
        cd "$tree/$r_cwd" || exit 97
        export CHIP_ARCH=$arch
        eval "$r_env_sh"
        # branch mode takes the lock here; patch mode already holds it on the inherited fd 9
        [ "$have_lock" = 1 ] || { exec 9>"$lock"; flock 9 || exit 97; }
        # a hung test from an earlier session leaves the card wedged (workers die at startup): reset first
        [ -n "${TT_DEVICE_LOCK:-}" ] || tt-smi -r 0 > /dev/null 2>&1 || true
        timeout --kill-after=60 "$r_timeout" bash -c "$r_cmd"
    ) > "$d/leg.log" 2>&1
    rc=$?
    kill "$beater" 2>/dev/null; wait "$beater" 2>/dev/null
    finish "$rc"
}
cmd_serve() {
    local D= once=0
    arch=; repo=; wroot=; applied=; interval=30
    while [ $# -gt 0 ]; do
        [ "$1" = --once ] && { once=1; shift; continue; }
        [ $# -ge 2 ] || die "$1 needs a value"
        case $1 in
        --run-dir) D=$2;; --arch) arch=$2;; --repo) repo=$2;; --worktree-root) wroot=$2;; --interval) interval=$2;;
        *) die "unknown arg $1";; esac; shift 2
    done
    [ -n "$D" ] && [ -n "$repo" ] || die "--run-dir and --repo are required"
    case $arch in wormhole|blackhole) ;; *) die "--arch must be wormhole|blackhole";; esac
    repo=$(readlink -f "$repo")
    git -C "$repo" rev-parse --show-toplevel > /dev/null 2>&1 || die "$repo is not a git checkout"
    wroot=${wroot:-$(dirname "$repo")}
    mkdir -p "$D/legs" || die "cannot create $D/legs"
    D=$(readlink -f "$D"); legs=$D/legs; slog=$legs/serve-$arch.log
    # a kill mid-leg must not leave the patch in --repo
    trap 'rc=$?; [ -n "${applied:-}" ] && git -C "$repo" apply -R "$applied" 2>/dev/null; exit $rc' EXIT
    trap 'exit 143' INT TERM
    log "serve start host=$(hostname) pid=$$ repo=$repo worktree-root=$wroot once=$once interval=$interval"
    while :; do
        if [ -e "$D/STOP" ] || [ -e "$legs/STOP-$arch" ]; then log "STOP present, exiting"; exit 0; fi
        beat
        claim_next
        if [ -n "$req" ]; then
            serve_one "$req"
            [ $once = 1 ] && exit 0
            continue
        fi
        [ $once = 1 ] && { log "nothing open for $arch"; exit 3; }
        sleep "$interval"
    done
}

cmd_wait() {
    local D= id= timeout=7200 t=0
    while [ $# -gt 0 ]; do case $1 in
        --run-dir) [ $# -ge 2 ] || die "$1 needs a value"; D=$2; shift 2;;
        --timeout) [ $# -ge 2 ] || die "$1 needs a value"; timeout=$2; shift 2;;
        -*) die "unknown arg $1";; *) id=$1; shift;; esac; done
    [ -n "$D" ] && [ -n "$id" ] || die "--run-dir and a request id are required"
    local d=$D/legs/$id
    [ -f "$d/request.json" ] || die "no request $id under $D/legs"
    while [ ! -f "$d/result.json" ]; do
        [ "$t" -ge "$timeout" ] && { echo "exit=125 log=$d/leg.log head= (no result after ${timeout}s)"; exit 125; }
        sleep 20; t=$((t + 20))
    done
    python3 -I - "$d/result.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print(f"exit={r['exit']} log={r['log']} head={r.get('head', '')}" + (f" message={r['message']}" if r.get('message') else ''))
sys.exit(int(r['exit']))
PY
}

cmd_status() {
    local D= interval=30
    while [ $# -gt 0 ]; do case $1 in
        --run-dir) [ $# -ge 2 ] || die "$1 needs a value"; D=$2; shift 2;;
        --interval) [ $# -ge 2 ] || die "$1 needs a value"; interval=$2; shift 2;;
        *) die "unknown arg $1";; esac; done
    [ -n "$D" ] || die "--run-dir is required"
    python3 -I - "$D/legs" "$interval" <<'PY'
import glob, json, os, re, sys, time
from datetime import datetime, timezone
def secs(iso):
    return int(time.time() - datetime.strptime(iso, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).timestamp())
for f in sorted(glob.glob(os.path.join(sys.argv[1], 'serve-*.alive'))):
    arch = os.path.basename(f)[6:-6]
    m = re.match(r'(\S+) pid=(\d+) (\S+) busy=(\S+)', open(f).read())
    if not m: print(f'serve {arch}: unreadable heartbeat {f}'); continue
    try: ago = secs(m[3])
    except Exception: print(f'serve {arch}: unreadable heartbeat {f}'); continue
    stale = ' STALE' if ago > 3 * int(sys.argv[2]) else ''
    print(f'serve {arch}: {m[1]} pid={m[2]} last seen {ago}s ago{stale} (busy={m[4]})')
def load(p):
    try: return json.load(open(p))
    except Exception: return None
def age(iso):
    try: s = secs(iso)
    except Exception: return '?'
    return f'{s}s' if s < 90 else f'{s // 60}m' if s < 5400 else f'{s // 3600}h' if s < 172800 else f'{s // 86400}d'
for f in sorted(glob.glob(os.path.join(sys.argv[1], '*', 'request.json'))):
    d = os.path.dirname(f); r = load(f) or {}; res = load(os.path.join(d, 'result.json'))
    server, ref = '-', r.get('created')
    if res:
        state, server, ref = f"done exit={res.get('exit')}", res.get('server', '-'), res.get('finished')
    elif os.path.exists(os.path.join(d, 'claimed.d')):
        state = 'claimed'
        try: server = open(os.path.join(d, 'claimed')).read().split()[0]
        except Exception: pass
    else:
        state = 'open'
    print(f"{os.path.basename(d)} {r.get('arch', '?')} {r.get('label', '?')} {state} "
          f"requester={r.get('requester', '?')} server={server} age={age(ref or '')}")
PY
}

case ${1:-} in
    request|serve|wait|status) sub=$1; shift; "cmd_$sub" "$@";;
    *) sed -n '2,7s/^# \{0,1\}//p' "$0" >&2; exit 2;;
esac
