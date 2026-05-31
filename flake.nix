{
  description = "TT-Metal - Tenstorrent's general compute framework";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfreePredicate = pkg:
            builtins.elem (pkgs.lib.getName pkg) [
              "tt-metal"
              "ttnn"
              "tt-metalium"
              "tenstorrent-firmware"
            ];
        };
      in {
        packages = {
          tt-metal = pkgs.callPackage ./nix/tt-metal.nix { };
          ttnn = pkgs.callPackage ./nix/ttnn.nix { };
          default = self.packages.${system}.tt-metal;
        };

        devShells.default = pkgs.callPackage ./shell.nix { };

        devShells.full = pkgs.mkShell {
          inputsFrom = [ self.devShells.${system}.default ];
          buildInputs = with pkgs; [
            clang-tools
            cmake
            gdb
            ninja
            python3Packages.debugpy
            python3Packages.pytest
            valgrind
          ];
          shellHook = ''
            echo "TT-Metal full development environment"
            echo "Run 'cmake --build build' to build"
            echo "Run 'pytest tests/ttnn/' to run tests"
          '';
        };
      });
}
