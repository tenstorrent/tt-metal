install_mpi_uflm(){
    DEB_URL="https://github.com/dmakoviichuk-tt/mpi-ulfm/releases/download/v5.0.7-ulfm/openmpi-ulfm_5.0.7-1_amd64.deb"
    DEB_FILE="$(basename "$DEB_URL")"

    # 1. Create temp workspace
    TMP_DIR="$(mktemp -d)"
    cleanup() { rm -rf "$TMP_DIR"; }
    trap cleanup EXIT INT TERM

    echo "→ Downloading $DEB_FILE …"
    wget -q --show-progress -O "$TMP_DIR/$DEB_FILE" "$DEB_URL"

    # 2. Install
    echo "→ Installing $DEB_FILE …"
    apt-get update -qq
    apt-get install -f -y "$TMP_DIR/$DEB_FILE"
}
install_mpi_uflm
