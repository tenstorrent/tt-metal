# default.nix — Non-flake entry point.
#
# For `nix-shell` (no flakes):
#   nix-shell default.nix
#
# For `nix-build`:
#   nix-build default.nix
#
# Flake users should use `flake.nix` directly instead.
#
# SPDX-License-Identifier: Apache-2.0

(import ./shell.nix { }).env
