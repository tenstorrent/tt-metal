#!/bin/bash
# Tenstorrent Blackhole Diagnostic Script
# Run this script on the host machine experiencing issues
# Usage: ./diagnose_blackhole.sh [--fix]

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

FIX_MODE=false
if [[ "$1" == "--fix" ]]; then
    FIX_MODE=true
    echo -e "${YELLOW}Running in FIX mode - will attempt to fix issues${NC}"
fi

echo "=============================================="
echo "  Tenstorrent Blackhole Diagnostic Script"
echo "  $(date)"
echo "=============================================="
echo ""

# Track issues found
ISSUES_FOUND=0

section() {
    echo ""
    echo -e "${BLUE}=== $1 ===${NC}"
}

ok() {
    echo -e "${GREEN}[OK]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    ((ISSUES_FOUND++)) || true
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    ((ISSUES_FOUND++)) || true
}

info() {
    echo -e "      $1"
}

# ============================================
# 1. System Information
# ============================================
section "System Information"
echo "Hostname: $(hostname)"
echo "Kernel: $(uname -r)"
echo "OS: $(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d'"' -f2 || echo "Unknown")"
echo "Architecture: $(uname -m)"

# ============================================
# 2. PCIe Device Detection
# ============================================
section "Tenstorrent PCIe Devices"

TT_DEVICES=$(lspci -d 1e52: 2>/dev/null | wc -l)
if [[ $TT_DEVICES -eq 0 ]]; then
    error "No Tenstorrent devices found!"
    info "Check if cards are properly seated in PCIe slots"
else
    ok "Found $TT_DEVICES Tenstorrent device(s)"
    lspci -d 1e52: -nn 2>/dev/null | while read line; do
        info "$line"
    done
fi

# Check for Blackhole specifically (device ID 401e)
BH_DEVICES=$(lspci -d 1e52:401e 2>/dev/null | wc -l)
WH_DEVICES=$(lspci -d 1e52:faca 2>/dev/null | wc -l)
GS_DEVICES=$(lspci -d 1e52:b140 2>/dev/null | wc -l)

echo ""
echo "Device breakdown:"
[[ $BH_DEVICES -gt 0 ]] && info "  Blackhole (401e): $BH_DEVICES"
[[ $WH_DEVICES -gt 0 ]] && info "  Wormhole (faca): $WH_DEVICES"
[[ $GS_DEVICES -gt 0 ]] && info "  Grayskull (b140): $GS_DEVICES"

# ============================================
# 3. Kernel Module (KMD) Status
# ============================================
section "Kernel Module (TT-KMD)"

if lsmod | grep -q tenstorrent; then
    ok "tenstorrent module is loaded"

    # Get KMD version
    KMD_VERSION=$(cat /sys/module/tenstorrent/version 2>/dev/null || modinfo tenstorrent 2>/dev/null | grep "^version:" | awk '{print $2}')
    if [[ -n "$KMD_VERSION" ]]; then
        info "KMD Version: $KMD_VERSION"

        # Version check for Blackhole
        if [[ $BH_DEVICES -gt 0 ]]; then
            MAJOR=$(echo $KMD_VERSION | cut -d'.' -f1)
            MINOR=$(echo $KMD_VERSION | cut -d'.' -f2)
            if [[ "$MAJOR" -ge 2 ]] && [[ "$MINOR" -ge 5 ]]; then
                ok "KMD version $KMD_VERSION is compatible with Blackhole (requires >= 2.5.0)"
            else
                error "KMD version $KMD_VERSION is too old for Blackhole (requires >= 2.5.0)"
            fi
        fi
    fi

    # Module parameters
    echo ""
    info "Module parameters:"
    for param in /sys/module/tenstorrent/parameters/*; do
        if [[ -r "$param" ]]; then
            info "  $(basename $param) = $(cat $param 2>/dev/null)"
        fi
    done
else
    error "tenstorrent module is NOT loaded"
    info "Try: sudo modprobe tenstorrent"

    if $FIX_MODE; then
        echo "Attempting to load module..."
        sudo modprobe tenstorrent && ok "Module loaded successfully" || error "Failed to load module"
    fi
fi

# Check /dev/tenstorrent
echo ""
if [[ -e /dev/tenstorrent ]]; then
    ok "/dev/tenstorrent exists"
    ls -la /dev/tenstorrent* 2>/dev/null | while read line; do
        info "$line"
    done
else
    error "/dev/tenstorrent does not exist"
fi

# ============================================
# 4. Virtualization Detection
# ============================================
section "Virtualization Environment"

IS_VM=false
VM_TYPE="none"

# Detect if running in a VM
if [[ -f /sys/class/dmi/id/product_name ]]; then
    PRODUCT=$(cat /sys/class/dmi/id/product_name 2>/dev/null)
    case "$PRODUCT" in
        *"Virtual Machine"*|*"VMware"*|*"VirtualBox"*|*"KVM"*|*"QEMU"*|*"Hyper-V"*)
            IS_VM=true
            VM_TYPE="$PRODUCT"
            ;;
    esac
fi

# Additional VM detection methods
if [[ -d /proc/xen ]] || grep -q "hypervisor" /proc/cpuinfo 2>/dev/null; then
    IS_VM=true
fi

if systemd-detect-virt --quiet 2>/dev/null; then
    IS_VM=true
    VM_TYPE=$(systemd-detect-virt 2>/dev/null || echo "unknown")
fi

if $IS_VM; then
    warn "Running inside a Virtual Machine: $VM_TYPE"
    info "VM passthrough mode requires specific configuration"
    echo ""

    # Check for vIOMMU
    VIOMMU_PRESENT=false
    if [[ -d /sys/devices/virtual/iommu ]] || dmesg 2>/dev/null | grep -qi "viommu\|vIOMMU"; then
        VIOMMU_PRESENT=true
        ok "vIOMMU appears to be present"
    else
        error "vIOMMU not detected - required for Blackhole in VM"
        info "Configure your hypervisor to enable vIOMMU for this VM"
        info "  - KVM/QEMU: Add <iommu model='intel'/> to VM XML"
        info "  - VMware: Enable 'Expose IOMMU to guest OS'"
    fi
else
    ok "Running on bare metal (not a VM)"
fi

# ============================================
# 5. IOMMU Configuration
# ============================================
section "IOMMU Configuration"

CMDLINE=$(cat /proc/cmdline)
echo "Kernel command line:"
info "$CMDLINE"
echo ""

# Check IOMMU settings
INTEL_IOMMU=$(echo "$CMDLINE" | grep -o "intel_iommu=[^ ]*" || echo "not set")
AMD_IOMMU=$(echo "$CMDLINE" | grep -o "amd_iommu=[^ ]*" || echo "not set")
IOMMU_PT=$(echo "$CMDLINE" | grep -o "iommu=pt" || echo "")

echo "IOMMU Settings:"
info "intel_iommu: $INTEL_IOMMU"
info "amd_iommu: $AMD_IOMMU"
info "iommu passthrough: ${IOMMU_PT:-not set}"

# Detect CPU vendor
CPU_VENDOR=$(grep -m1 "vendor_id" /proc/cpuinfo | awk '{print $3}')
info "CPU Vendor: $CPU_VENDOR"

# Check if IOMMU is properly enabled
if [[ "$CPU_VENDOR" == "GenuineIntel" ]]; then
    if [[ "$INTEL_IOMMU" != *"on"* ]]; then
        warn "Intel IOMMU is not enabled. For Blackhole TLB support, add 'intel_iommu=on' to kernel parameters"
        info "Edit /etc/default/grub, add to GRUB_CMDLINE_LINUX_DEFAULT, run 'sudo update-grub' and reboot"
    else
        ok "Intel IOMMU is enabled"
    fi
elif [[ "$CPU_VENDOR" == "AuthenticAMD" ]]; then
    if [[ "$AMD_IOMMU" != *"on"* ]]; then
        warn "AMD IOMMU is not enabled. For Blackhole TLB support, add 'amd_iommu=on' to kernel parameters"
        info "Edit /etc/default/grub, add to GRUB_CMDLINE_LINUX_DEFAULT, run 'sudo update-grub' and reboot"
    else
        ok "AMD IOMMU is enabled"
    fi
fi

# Check IOMMU groups
echo ""
IOMMU_GROUPS=$(ls -d /sys/kernel/iommu_groups/*/ 2>/dev/null | wc -l)
if [[ $IOMMU_GROUPS -gt 0 ]]; then
    ok "IOMMU groups present: $IOMMU_GROUPS groups"

    # Check if TT devices have IOMMU groups
    echo ""
    info "Tenstorrent devices in IOMMU groups:"
    for dev in $(lspci -d 1e52: -D 2>/dev/null | awk '{print $1}'); do
        GROUP=$(find /sys/kernel/iommu_groups/*/devices -name "$dev" 2>/dev/null | grep -o "iommu_groups/[0-9]*" | head -1)
        if [[ -n "$GROUP" ]]; then
            info "  $dev -> $GROUP"
        else
            warn "  $dev -> No IOMMU group (IOMMU may not be fully enabled)"
        fi
    done
else
    warn "No IOMMU groups found - IOMMU may not be properly enabled"
    if $IS_VM; then
        error "In a VM, this likely means vIOMMU is not configured on the hypervisor"
    fi
fi

# ============================================
# 5. Hugepage Configuration
# ============================================
section "Hugepage Configuration"

echo "Memory info (hugepages):"
grep -i huge /proc/meminfo | while read line; do
    info "$line"
done

echo ""

# 2MB Hugepages
HP_2M_TOTAL=$(cat /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages 2>/dev/null || echo "0")
HP_2M_FREE=$(cat /sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages 2>/dev/null || echo "0")
info "2MB Hugepages: Total=$HP_2M_TOTAL, Free=$HP_2M_FREE"

# 1GB Hugepages
HP_1G_TOTAL=$(cat /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages 2>/dev/null || echo "0")
HP_1G_FREE=$(cat /sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages 2>/dev/null || echo "0")
info "1GB Hugepages: Total=$HP_1G_TOTAL, Free=$HP_1G_FREE"

# Check hugepage mounts
echo ""
echo "Hugepage mounts:"
mount | grep huge | while read line; do
    info "$line"
done

# Check /dev/hugepages
echo ""
if [[ -d /dev/hugepages ]]; then
    ok "/dev/hugepages exists"
    HP_FILES=$(ls -la /dev/hugepages/ 2>/dev/null | grep -c tenstorrent || echo "0")
    info "Tenstorrent hugepage files in /dev/hugepages: $HP_FILES"
else
    warn "/dev/hugepages does not exist"
fi

# Check /dev/hugepages-1G (1GB hugepages for Blackhole)
if [[ -d /dev/hugepages-1G ]]; then
    ok "/dev/hugepages-1G exists"
    HP_1G_FILES=$(ls /dev/hugepages-1G/ 2>/dev/null | grep -c tenstorrent || echo "0")
    info "Tenstorrent 1GB hugepage files: $HP_1G_FILES"

    if [[ $BH_DEVICES -gt 0 ]]; then
        # Blackhole typically needs 4 channels per device
        EXPECTED_FILES=$((BH_DEVICES * 4))
        if [[ $HP_1G_FILES -lt $EXPECTED_FILES ]]; then
            warn "Expected at least $EXPECTED_FILES 1GB hugepage files for $BH_DEVICES Blackhole device(s)"
        fi
    fi

    ls -la /dev/hugepages-1G/ 2>/dev/null | head -20 | while read line; do
        info "$line"
    done
else
    if [[ $BH_DEVICES -gt 0 ]]; then
        warn "/dev/hugepages-1G does not exist (recommended for Blackhole)"
        info "To create: sudo mkdir /dev/hugepages-1G && sudo mount -t hugetlbfs -o pagesize=1G none /dev/hugepages-1G"
    fi
fi

# Hugepage requirements check
echo ""
if [[ $BH_DEVICES -gt 0 ]]; then
    REQUIRED_1G_PAGES=$((BH_DEVICES * 4))  # 4 x 1GB pages per device
    if [[ $HP_1G_TOTAL -lt $REQUIRED_1G_PAGES ]]; then
        warn "Insufficient 1GB hugepages for $BH_DEVICES Blackhole device(s)"
        info "Required: $REQUIRED_1G_PAGES, Available: $HP_1G_TOTAL"
        info "To allocate: echo $REQUIRED_1G_PAGES | sudo tee /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages"
        info "For persistent config, add to /etc/sysctl.conf: vm.nr_hugepages_1GB = $REQUIRED_1G_PAGES"
    else
        ok "Sufficient 1GB hugepages for Blackhole ($HP_1G_TOTAL >= $REQUIRED_1G_PAGES)"
    fi
fi

# ============================================
# 6. Firmware Version Check
# ============================================
section "Firmware Version"

# Try to get firmware version using tt-smi if available
if command -v tt-smi &> /dev/null; then
    ok "tt-smi is available"
    echo ""
    echo "Device firmware information:"
    tt-smi 2>/dev/null | grep -E "(firmware|version|FW)" -i | head -20 | while read line; do
        info "$line"
    done

    # Full tt-smi output
    echo ""
    info "Full tt-smi output:"
    tt-smi 2>/dev/null | head -50 | while read line; do
        info "$line"
    done
else
    warn "tt-smi not found in PATH"
    info "Install tt-smi for firmware diagnostics: pip install tt-smi or from https://github.com/tenstorrent/tt-smi"
fi

# Check for tt-flash
echo ""
if command -v tt-flash &> /dev/null; then
    ok "tt-flash is available"
    info "tt-flash version: $(tt-flash --version 2>/dev/null || echo 'unknown')"
else
    warn "tt-flash not found in PATH"
fi

# ============================================
# 7. TLB Resource Check
# ============================================
section "TLB Resources"

info "Blackhole TLB specifications:"
info "  - 202x 2 MiB TLB windows"
info "  - 8x 4 GiB TLB windows"
echo ""

# Check for processes using tenstorrent devices
TT_PROCS=$(lsof /dev/tenstorrent* 2>/dev/null | grep -v "^COMMAND" | wc -l || echo "0")
if [[ $TT_PROCS -gt 0 ]]; then
    warn "Found $TT_PROCS process(es) using Tenstorrent devices"
    info "This may cause TLB resource exhaustion"
    echo ""
    info "Processes:"
    lsof /dev/tenstorrent* 2>/dev/null | head -20 | while read line; do
        info "  $line"
    done

    if $FIX_MODE; then
        echo ""
        warn "FIX mode: Consider killing these processes and reloading the kernel module"
    fi
else
    ok "No processes currently using Tenstorrent devices"
fi

# ============================================
# 8. Kernel Messages (if accessible)
# ============================================
section "Recent Kernel Messages"

# Try to read dmesg
if dmesg 2>/dev/null | tail -1 > /dev/null 2>&1; then
    echo "Recent tenstorrent-related kernel messages:"
    dmesg 2>/dev/null | grep -i "tenstorrent\|iommu.*1e52\|pci.*1e52" | tail -20 | while read line; do
        info "$line"
    done

    echo ""
    echo "Recent errors in kernel log:"
    dmesg 2>/dev/null | grep -iE "(error|fail|warn).*tenstorrent" | tail -10 | while read line; do
        info "$line"
    done
else
    warn "Cannot read kernel messages (need root or dmesg permission)"
    info "Run with sudo for full diagnostics"

    # Try journalctl as alternative
    if command -v journalctl &> /dev/null; then
        echo ""
        info "Trying journalctl..."
        journalctl -k --no-pager 2>/dev/null | grep -i "tenstorrent" | tail -10 | while read line; do
            info "$line"
        done
    fi
fi

# ============================================
# 9. Python Environment Check
# ============================================
section "Python Environment"

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    ok "Python3 available: $PYTHON_VERSION"

    # Check for ttnn
    if python3 -c "import ttnn" 2>/dev/null; then
        ok "ttnn module is importable"
        TTNN_VERSION=$(python3 -c "import ttnn; print(ttnn.__version__)" 2>/dev/null || echo "unknown")
        info "ttnn version: $TTNN_VERSION"
    else
        warn "ttnn module not found or not importable"
    fi
else
    warn "Python3 not found"
fi

# ============================================
# 11. Summary and Recommendations
# ============================================
section "Summary"

if [[ $ISSUES_FOUND -eq 0 ]]; then
    echo -e "${GREEN}No issues detected!${NC}"
else
    echo -e "${YELLOW}Found $ISSUES_FOUND potential issue(s)${NC}"
    echo ""

    if $IS_VM; then
        echo "============================================"
        echo "  VIRTUAL MACHINE SPECIFIC FIXES"
        echo "============================================"
        echo ""
        echo "For Blackhole in a VM, you need BOTH host and guest configuration:"
        echo ""
        echo "=== ON THE HOST SERVER ==="
        echo ""
        echo "1. Enable IOMMU in BIOS/UEFI (Intel VT-d or AMD-Vi)"
        echo ""
        echo "2. Host kernel parameters (/etc/default/grub):"
        if [[ "$CPU_VENDOR" == "GenuineIntel" ]]; then
            echo "   GRUB_CMDLINE_LINUX_DEFAULT=\"intel_iommu=on iommu=pt\""
        else
            echo "   GRUB_CMDLINE_LINUX_DEFAULT=\"amd_iommu=on iommu=pt\""
        fi
        echo ""
        echo "3. Configure PCIe passthrough for device 1e52:401e"
        echo "   - KVM: Use vfio-pci driver binding"
        echo "   - VMware: Enable DirectPath I/O"
        echo ""
        echo "4. Enable vIOMMU for the VM:"
        echo "   - KVM/libvirt: Add to VM XML:"
        echo "     <iommu model='intel'>"
        echo "       <driver intremap='on' caching_mode='on'/>"
        echo "     </iommu>"
        echo "   - VMware: Enable 'Expose IOMMU to guest OS'"
        echo ""
        echo "=== INSIDE THE VM (Guest) ==="
        echo ""
    fi

    echo "Common fixes for TLB allocation errors:"
    echo ""
    echo "1. Enable IOMMU properly (in guest kernel if VM):"
    if [[ "$CPU_VENDOR" == "GenuineIntel" ]]; then
        echo "   Add to /etc/default/grub GRUB_CMDLINE_LINUX_DEFAULT:"
        echo "   intel_iommu=on iommu=pt"
    else
        echo "   Add to /etc/default/grub GRUB_CMDLINE_LINUX_DEFAULT:"
        echo "   amd_iommu=on iommu=pt"
    fi
    echo "   Then: sudo update-grub && sudo reboot"
    echo ""
    echo "2. Reload kernel module (clears TLB resources):"
    echo "   sudo rmmod tenstorrent && sudo modprobe tenstorrent"
    echo ""
    echo "3. Ensure sufficient 1GB hugepages:"
    echo "   echo 16 | sudo tee /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages"
    echo ""
    echo "4. Kill any orphaned processes using the device:"
    echo "   lsof /dev/tenstorrent* | awk 'NR>1 {print \$2}' | xargs -r kill"
    echo ""

    if $IS_VM; then
        echo "5. Verify vIOMMU is working:"
        echo "   ls /sys/kernel/iommu_groups/"
        echo "   (Should show numbered directories if vIOMMU is active)"
        echo ""
    fi
fi

# ============================================
# Save output to file
# ============================================
REPORT_FILE="/tmp/tt_blackhole_diagnostic_$(date +%Y%m%d_%H%M%S).txt"
echo ""
echo "=============================================="
echo "Diagnostic complete. Re-run with 'bash -x' for debug output."
echo "To save this report, run:"
echo "  $0 2>&1 | tee $REPORT_FILE"
echo "=============================================="

exit $ISSUES_FOUND
