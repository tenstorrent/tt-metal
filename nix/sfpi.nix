{
  lib,
  stdenv,
  fetchurl,
  runCommand,
  autoPatchelfHook,
  ncurses,
  isl_0_23,
  mpfr,
  libmpc,
  xz,
  zstd,
  expat,
}:

let
  versionFile = ../tt_metal/sfpi-version;
  versionLines = lib.splitString "\n" (builtins.readFile versionFile);
  readVersionValue =
    name:
    let
      prefix = "${name}='";
      line =
        lib.findFirst (lib.hasPrefix prefix) (throw "Missing ${name} in ${toString versionFile}")
          versionLines;
    in
    lib.removeSuffix "'" (lib.removePrefix prefix line);

  version = readVersionValue "sfpi_version";
  repository = readVersionValue "sfpi_repo";
  sources = {
    aarch64-linux = {
      archiveArch = "aarch64";
      sha256 = readVersionValue "sfpi_aarch64_debian_txz_hash";
    };
    x86_64-linux = {
      archiveArch = "x86_64";
      sha256 = readVersionValue "sfpi_x86_64_debian_txz_hash";
    };
  };
  source =
    sources.${stdenv.hostPlatform.system}
      or (throw "SFPI does not support ${stdenv.hostPlatform.system}");
in
runCommand "sfpi-${version}"
  {
    inherit version;

    nativeBuildInputs = [
      autoPatchelfHook
    ];

    buildInputs = [
      ncurses
      mpfr
      libmpc
      xz
      zstd
      expat
    ];

    src = fetchurl {
      url = "${repository}/releases/download/${version}/sfpi_${version}_${source.archiveArch}_debian.txz";
      sha256 = source.sha256;
    };
  }
  ''
    runPhase unpackPhase
    mkdir -p "$out"
    cp -r ../"$sourceRoot" "$out/sfpi"
    runPhase fixupPhase
  ''
