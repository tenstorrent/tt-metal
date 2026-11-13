#!/bin/bash
set -e

# The image ships no SSH host keys (they would be shared by every container);
# generate this container's on first start. Existing keys are kept.
ssh-keygen -A

# Start SSHD in the background
service ssh start

# Exec the passed command (replace shell with target command)
exec "$@"
