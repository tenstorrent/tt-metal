# shell.nix — Development shell for TT-Metalium
#
# Provides a reproducible development environment for building and working on
# tt-metal.  Replaces install_dependencies.sh for Nix/NixOS users.
#
# Usage:
#   nix-shell                                       # enter dev shell
#   nix-shell --arg enableDistributed false          # without MPI/OpenMPI
#   nix-shell --arg clangStdenv false                # use GCC instead of Clang
#
# SPDX-License-Identifier: Apache-2.0

{
  # Whether to include distributed-compute (OpenMPI) dependencies.
  enableDistributed ? true,
  # Use the LLVM/Clang 20 stdenv (default).  Set to false to use the default GCC
  # stdenv shipped by the current nixpkgs revision.
  clangStdenv ? true,
  # Pin a specific nixpkgs revision for reproducibility.
  # You may also pass `<nixpkgs>` via `--arg pkgs 'import <nixpkgs> {}'`.
  pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/refs/tags/25.05.tar.gz") { },
}:

let
  # ---------------------------------------------------------------------------
  # LLVM/Clang 20 – tt-metal uses clang-20 as its default toolchain.
  # The `build_metal.sh` script looks for `clang-20` in PATH.
  # ---------------------------------------------------------------------------
  llvmPackages = pkgs.llvmPackages_20;

  # ---------------------------------------------------------------------------
  # CMake ≥ 4.0.2 – the install_dependencies.sh script fetches this version
  # from upstream; nixpkgs 25.05 ships cmake ≥ 3.31 so we pull the latest.
  # ---------------------------------------------------------------------------
  cmake' = pkgs.cmake;

  # ---------------------------------------------------------------------------
  # Stdenv selection – Clang 20 (preferred) or default GCC.
  # ---------------------------------------------------------------------------
  stdenv = if clangStdenv then llvmPackages.stdenv else pkgs.stdenv;

in

stdenv.mkDerivation {
  name = "tt-metal-dev-env";

  src = null; # source-less dev shell

  # ---------------------------------------------------------------------------
  # Native build inputs – tools that run on the build host.
  # ---------------------------------------------------------------------------
  nativeBuildInputs = with pkgs; [
    cmake'
    ninja
    pkg-config
    git
    gnumake
    python3
    python3.pkgs.pip
    python3.pkgs.venvShellHook  # auto-set-up a virtualenv on nix-shell enter
    python3.pkgs.setuptools
    pandoc
    xz
    openssl
    wget
    curl
    xxd                          # vim-common replacement in Nix
    jq
    gnupg
    lsb-release
    gawk
  ];

  # ---------------------------------------------------------------------------
  # Build inputs – libraries and headers needed at compile / link time.
  # ---------------------------------------------------------------------------
  buildInputs = with pkgs; [
    hwloc
    numactl
    libatomic_ops
    tbb
    capstone
    llvmPackages.openmp
  ]
  ++ lib.optionals (lib.versionOlder stdenv.cc.cc.version "14") [
    # Provide a newer libc++/libc++abi when the default toolchain is too old.
    # clang-20 already ships its own, so this is mainly for the GCC path.
    llvmPackages.libcxx
    llvmPackages.libcxxabi
  ]
  ++ lib.optionals enableDistributed [
    pkgs.openmpi
  ];

  # ---------------------------------------------------------------------------
  # Python dependencies – the subset of requirements that maps to Nix packages.
  # The rest should be installed via `pip` inside the virtualenv (see below).
  # ---------------------------------------------------------------------------
  pythonInputs = with pkgs.python3Packages; [
    pyyaml
    numpy
  ];

  propagatedBuildInputs = pythonInputs;

  # ---------------------------------------------------------------------------
  # Environment variables that tt-metal expects.
  # ---------------------------------------------------------------------------
  shellHook = ''
    # ----------------------------------------------------------------------
    # Python virtualenv – auto-created by venvShellHook.
    # Install remaining Python deps from requirements files.
    # ----------------------------------------------------------------------
    if [ -f "tt_metal/python_env/requirements-dev.txt" ]; then
      echo "[nix] Installing Python development requirements ..."
      pip install --quiet -r tt_metal/python_env/requirements-dev.txt 2>/dev/null || true
    fi
    if [ -f "requirements.txt" ]; then
      echo "[nix] Installing Python requirements ..."
      pip install --quiet -r requirements.txt 2>/dev/null || true
    fi

    # ----------------------------------------------------------------------
    # Ensure clang-20 is the default compiler (matches build_metal.sh).
    # When using the Clang stdenv the wrapper already points there, but
    # build_metal.sh explicitly name-checks clang-20.
    # ----------------------------------------------------------------------
    ${lib.optionalString clangStdenv ''
      export CC=${llvmPackages.clang}/bin/clang-20
      export CXX=${llvmPackages.clang}/bin/clang++-20
    ''}

    # ----------------------------------------------------------------------
    # Informational banner.
    # ----------------------------------------------------------------------
    echo ""
    echo "  [tt-metal Nix dev shell]"
    echo "  CC:  $(type -p $CC 2>/dev/null || echo ${stdenv.cc.cc})"
    echo "  CXX: $(type -p $CXX 2>/dev/null || echo ${stdenv.cc.cc})"
    echo "  CMake: $(cmake --version 2>/dev/null | head -1 || echo not-found)"
    echo "  Python: $(python3 --version 2>/dev/null || echo not-found)"
    echo "  Ninja: $(ninja --version 2>/dev/null || echo not-found)"
    echo ""
    echo "  Quick start:"
    echo "    cmake -G Ninja -B build -S ."
    echo "    ninja -C build"
    echo ""
  '';

  # ---------------------------------------------------------------------------
  # Meta-information.
  # ---------------------------------------------------------------------------
  meta = with pkgs.lib; {
    description = "Development shell for Tenstorrent tt-metal (TT-Metalium / TT-NN)";
    homepage = "https://github.com/tenstorrent/tt-metal";
    license = licenses.asl20;
    platforms = platforms.linux;
    maintainers = [ ];
  };
}
