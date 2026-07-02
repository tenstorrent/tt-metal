# TT-Metal Nix Support

## Quick Start

```bash
# Using shell.nix
nix-shell nix/shell.nix

# Using flake.nix
nix develop
```

## Files
- `shell.nix` — Development environment with all build dependencies
- `flake.nix` — Reproducible Nix Flake for the full TT stack

## NixOS Notes
On NixOS, the FHS is not present. The shell.nix automatically sets `TT_HOME` to the current directory.
