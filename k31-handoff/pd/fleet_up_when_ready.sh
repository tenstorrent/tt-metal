#!/bin/bash
# Bring the KVM fleet up once both tables newer than $1 (epoch s) exist.
since=$1; P=/data/bklockiewicz/k31run/pd/prefill_table.pb D=/data/bklockiewicz/k31run/pd/decode_table.pb
until [ -f $P ] && [ -f $D ] && [ $(stat -c %Y $P) -gt $since ] && [ $(stat -c %Y $D) -gt $since ]; do sleep 10; done
echo "both tables at $(date -u +%T); fleet up"
export PATH=/data/bklockiewicz/k31-migration/python_env/bin:$PATH
cd /data/bklockiewicz/tt-d-gen && ./kv_manager/scripts/fleet/launch_kvm_fleet.sh --config /data/bklockiewicz/k31run/pd/fleet-k31-pd.conf up 2>&1 | grep -E "OK|FAIL|complete|ERROR"
