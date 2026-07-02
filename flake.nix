{
  description = "A reproducible development environment for the Tenstorrent stack";

  inputs = {
    # Pinning to a specific nixpkgs commit ensures reproducibility.
    # Users can update this by running `nix flake update`.
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    # Use flake-utils to provide outputs for common systems (x86_64-linux, aarch64-linux, etc.)
    flake-utils.lib.eachDefaultSystem (system:
      let
        # Import nixpkgs for the given system.
        # `allowUnfree = true` might be necessary for certain drivers or software.
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        # --- Python Environment ---
        # Defines a Python environment with all necessary packages.
        # This is more reliable than using a requirements.txt with pip.
        pythonEnv = pkgs.python311.withPackages (ps: with ps; [
          # Core ML/data science
          numpy
          pyyaml

          # It's recommended to use the PyTorch from nixpkgs.
          # For CUDA-enabled PyTorch, you would need to configure nixpkgs with CUDA support.
          torch

          # Development and testing tools
          pytest
          ruff
          black
          
          # General utilities
          pip
          setuptools
          wheel
          tqdm
        ]);

        # --- C++ Dependencies ---
        # A list of all system dependencies required to build the C++ components.
        cppDeps = with pkgs; [
          # Compilers and linkers
          llvmPackages_latest.clang
          llvmPackages_latest.lld
          
          # Build systems
          cmake
          ninja
          pkg-config

          # Core libraries
          boost
          yaml-cpp
          spdlog

          # Testing libraries
          gtest # for unit tests
          gmock # for mocking

          # Debugging and analysis tools
          gdb
          valgrind
        ];

      in
      {
        # --- Packages ---
        # This defines how to build the software in this repository.
        # `nix build .` will build the default package.
        packages.default = pkgs.stdenv.mkDerivation {
          pname = "tt-metal";
          version = "0.1.0"; # Or derive from git
          src = ./.;

          nativeBuildInputs = with pkgs; [
            cmake
            ninja
            pkg-config
          ];

          buildInputs = cppDeps ++ [ pythonEnv ];

          # NOTE: The build steps below are placeholders.
          # They should be adapted to the project's actual build process.
          # For example, you might need to pass specific CMake flags.
          configurePhase = ''
            runHook preConfigure
            cmake -S . -B build -G Ninja \
              -DCMAKE_BUILD_TYPE=Release \
              -DCMAKE_INSTALL_PREFIX=$out
            runHook postConfigure
          '';
          
          buildPhase = ''
            runHook preBuild
            cmake --build build
            runHook postBuild
          '';

          installPhase = ''
            runHook preInstall
            cmake --install build
            runHook postInstall
          '';
        };

        # --- Development Shell ---
        # This provides a development environment with all dependencies.
        # Enter the shell by running `nix develop`.
        # This replaces scripts like `install_dependencies.sh`.
        devShells.default = pkgs.mkShell {
          name = "tt-dev-shell";

          # Use packages from our `packages.default` definition.
          # This includes all `buildInputs` and `nativeBuildInputs`.
          inputsFrom = [ self.packages.${system}.default ];

          # Add any extra tools needed for development but not for building.
          packages = with pkgs; [
            git
            nixpkgs-fmt # For formatting .nix files
          ];

          # This hook runs when you enter the shell.
          # Use it to set up environment variables and print welcome messages.
          shellHook = ''
            echo "
            ===========================================================
            Welcome to the Tenstorrent Development Shell!

            - C/C++ toolchain and libraries are in your PATH.
            - Python environment with all dependencies is available.
              - Python interpreter: ${pythonEnv.interpreter}/bin/python
            - Run 'exit' or press Ctrl+D to leave.
            ===========================================================
            "

            # Set environment variables required by the project's build system or scripts.
            export TT_METAL_HOME=$(pwd)
            export PYTHONPATH="${pythonEnv}/${pythonEnv.sitePackages}:$PYTHONPATH"

            # You can define aliases for common commands.
            alias build="cmake --build build"
            alias test="ctest --test-dir build"
          '';
        };

        # --- Formatter ---
        # Defines a code formatter for the repository.
        # Run with `nix fmt`.
        formatter.default = pkgs.nixpkgs-fmt;

      });
}
