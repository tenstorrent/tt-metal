#!/bin/bash
set -e

# Start SSHD in the background. The image ships no SSH host keys; the
# /usr/sbin/sshd wrapper generates this container's on start.
service ssh start

# Exec the passed command (replace shell with target command)
exec "$@"
