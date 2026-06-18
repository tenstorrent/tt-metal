{
  description = "Tenstorrent TT-Metal development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          name = "tt-metal-dev";
          buildInputs = with pkgs; [
            cmake gcc gnumake pkg-config
            python3 python3Packages.pip python3Packages.virtualenv
            libffi openssl zlib
            git gdb valgrind
            doxygen graphviz
          ];
          shellHook = ''
            echo "🖥️  tt-metal dev environment (Nix Flake)"
            if [ ! -d .venv ]; then
              python3 -m venv .venv
            fi
            source .venv/bin/activate
            export TT_HOME=$(pwd)
          '';
        };
      }
    );
}
