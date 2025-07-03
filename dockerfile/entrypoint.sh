#!/bin/bash
set -e

# Start SSHD in the background
service ssh start

# Exec the passed command (replace shell with target command)
exec "$@"
