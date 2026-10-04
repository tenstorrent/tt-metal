#!/bin/bash
# Stop the stack started by start.sh: SIGTERM -> the front end stops the engine daemon (rt.stop(), then the runner's
# shutdown sentinel), the runner closes the mesh; run_safe_pytest then releases the device lock.
ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
DIR=${XING_SERVE_DIR:-$ROOT/generated/xing_serve}
[ -f "$DIR/server.pid" ] || { echo "no pid file $DIR/server.pid (not running?)"; exit 1; }
kill -TERM "$(cat "$DIR/server.pid")" && echo "sent SIGTERM to $(cat "$DIR/server.pid"); watch $DIR/server.log for 'XING_SERVE: stopped'"
