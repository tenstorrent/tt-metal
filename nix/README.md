# Nix packaging for tt-metal

This directory provides Nix/NixOS support for the Tenstorrent TT stack
(TT-Metalium + TT-NN).

## Quick start (development shell)

``` sh
# With flakes:
nix develop

# Without flakes:
nix-shell
```

This gives you a shell with all build dependencies from
`install_dependencies.sh`: Clang 20, CMake, Ninja, hwloc, libnuma, TBB,
Capstone, OpenMPI, Python with pip/venv, and more.

Once inside the shell:

``` sh
cmake -G Ninja -B build -S .
ninja -C build
```

## Build tt-metal as a Nix package

``` sh
# With flakes:
nix build .#tt-metal

# Without flakes:
nix-build -A tt-metal
```

## Using in a NixOS configuration

Add the flake as an input and apply the overlay:

``` nix
# flake.nix
{
  inputs.tt-metal-nix.url = "github:tenstorrent/tt-metal";

  outputs = { self, nixpkgs, tt-metal-nix, ... }: {
    nixosConfigurations.my-machine = nixpkgs.lib.nixosSystem {
      modules = [
        { nixpkgs.overlays = [ tt-metal-nix.overlays.default ]; }
        # Now `pkgs.tt-metal` is available.
      ];
    };
  };
}
```

## Files

| File          | Purpose                                              |
|---------------|------------------------------------------------------|
| `flake.nix`   | Flake providing package, devShell, and overlay       |
| `shell.nix`   | Development shell (called by `flake.nix` internally) |
| `default.nix` | Non-flake compat entry point                         |

## Notes for NixOS users

NixOS does not follow the Filesystem Hierarchy Standard (FHS), so paths
like `/opt/tenstorrent` are not available.  This flake handles the SFPI
firmware package by extracting the Tenstorrent `.deb` release and making
its contents available via Nix store paths.

The Tenstorrent kernel module (TT-KMD) and firmware flash tool are not
packaged here — they should be installed via the distribution's package
manager or the official TT-Installer script.
