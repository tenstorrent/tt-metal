{ lib, stdenv, cmake, ninja, python3, pkg-config, openmpi, hwloc, numactl
, gflags, glog, abseil-cpp, libxml2, libyaml, fetchFromGitHub, git
, which, gcc14
}:

stdenv.mkDerivation (finalAttrs: {
  pname = "tt-metal";
  version = "0.58.0";

  src = builtins.path { path = ./..; name = "tt-metal-source"; };

  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
    python3
    python3.pkgs.pybind11
    git
    which
  ];

  buildInputs = [
    openmpi
    hwloc
    numactl
    gflags
    glog
    abseil-cpp
    libxml2
    libyaml
    gcc14.cc.lib
    python3.pkgs.numpy
    python3.pkgs.loguru
    python3.pkgs.networkx
    python3.pkgs.pyyaml
    python3.pkgs.click
  ];

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
    "-DBUILD_SHARED_LIBS=ON"
  ];

  enableParallelBuilding = true;

  meta = with lib; {
    description = "TT-Metal: General compute framework for Tenstorrent devices";
    homepage = "https://github.com/tenstorrent/tt-metal";
    license = licenses.asl20;
    maintainers = with maintainers; [ ];
    platforms = [ "x86_64-linux" ];
  };
})
