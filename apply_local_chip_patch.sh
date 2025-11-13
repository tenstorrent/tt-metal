#!/bin/bash
# Script to apply local_chip.cpp patch to release CHIP_IN_USE lock after initialization

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UMD_DIR="${SCRIPT_DIR}/tt_metal/third_party/umd"
TARGET_FILE="${UMD_DIR}/device/chip/local_chip.cpp"

echo "Applying local_chip.cpp patch to release CHIP_IN_USE lock after initialization..."

cd "${UMD_DIR}"

# Check if file exists
if [ ! -f "${TARGET_FILE}" ]; then
    echo "Error: ${TARGET_FILE} not found!"
    exit 1
fi

# Apply the patch using git apply or sed
cat > /tmp/local_chip.patch << 'EOF'
--- a/device/chip/local_chip.cpp
+++ b/device/chip/local_chip.cpp
@@ -190,14 +190,20 @@ void LocalChip::start_device() {
         return;
     }

-    // TODO: acquire mutex should live in Chip class. Currently we don't have unique id for all chips.
-    // The lock here should suffice since we have to open Local chip to have Remote chips initialized.
-    chip_started_lock_.emplace(acquire_mutex(MutexType::CHIP_IN_USE, tt_device_->get_pci_device()->get_device_num()));
+    // Acquire CHIP_IN_USE lock to prevent concurrent initialization, but release it after initialization completes.
+    // This allows multiple processes to query device info and use the chip after initialization.
+    {
+        auto chip_init_lock = acquire_mutex(MutexType::CHIP_IN_USE, tt_device_->get_pci_device()->get_device_num());

-    check_pcie_device_initialized();
-    sysmem_manager_->pin_or_map_sysmem_to_device();
-    if (!tt_device_->get_pci_device()->is_mapping_buffer_to_noc_supported()) {
-        // If this is supported by the newer KMD, UMD doesn't have to program the iatu.
-        init_pcie_iatus();
+        check_pcie_device_initialized();
+        sysmem_manager_->pin_or_map_sysmem_to_device();
+        if (!tt_device_->get_pci_device()->is_mapping_buffer_to_noc_supported()) {
+            // If this is supported by the newer KMD, UMD doesn't have to program the iatu.
+            init_pcie_iatus();
+        }
+        initialize_membars();
+        // Lock is automatically released when chip_init_lock goes out of scope here
     }
-    initialize_membars();
 }
EOF

# Try to apply with git
if git apply --check /tmp/local_chip.patch 2>/dev/null; then
    git apply /tmp/local_chip.patch
    echo "✅ Patch applied successfully using git apply"
else
    echo "Git apply failed, using manual patching..."

    # Manual patching using sed
    sed -i '
    # Find the start_device function section
    /^void LocalChip::start_device() {/,/^}/ {
        # Replace the mutex acquisition and initialization block
        s|    chip_started_lock_.emplace(acquire_mutex(MutexType::CHIP_IN_USE, tt_device_->get_pci_device()->get_device_num()));|    // Acquire CHIP_IN_USE lock to prevent concurrent initialization, but release it after initialization completes.\n    // This allows multiple processes to query device info and use the chip after initialization.\n    {\n        auto chip_init_lock = acquire_mutex(MutexType::CHIP_IN_USE, tt_device_->get_pci_device()->get_device_num());|

        # Add closing brace and comment after initialize_membars
        /^    initialize_membars();$/a\        // Lock is automatically released when chip_init_lock goes out of scope here\n    }

        # Indent the initialization calls
        s|^    check_pcie_device_initialized();$|        check_pcie_device_initialized();|
        s|^    sysmem_manager_->pin_or_map_sysmem_to_device();$|        sysmem_manager_->pin_or_map_sysmem_to_device();|
        s|^    if (!tt_device_->get_pci_device()->is_mapping_buffer_to_noc_supported()) {$|        if (!tt_device_->get_pci_device()->is_mapping_buffer_to_noc_supported()) {|
        s|^        // If this is supported by the newer KMD, UMD doesn'\''t have to program the iatu.$|            // If this is supported by the newer KMD, UMD doesn'\''t have to program the iatu.|
        s|^        init_pcie_iatus();$|            init_pcie_iatus();|
        s|^    }$|        }|
        s|^    initialize_membars();$|        initialize_membars();|
    }
    ' "${TARGET_FILE}"

    echo "✅ Patch applied successfully using sed"
fi

rm -f /tmp/local_chip.patch

# Show the changes
echo ""
echo "Changes made to ${TARGET_FILE}:"
git diff device/chip/local_chip.cpp | head -50

echo ""
echo "✅ Done! The CHIP_IN_USE lock will now be released after device initialization."
echo "   This allows multiple processes (like tt-smi) to query device info."
