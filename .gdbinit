# tt-metal GDB configuration

# Load tt-metal pretty printers
source /localdev/fbajraktari/tt-metal/ttmetal-gdb.py

# Pretty printing settings
set print pretty on
set print object on
set print static-members on
set print vtbl on
set print demangle on
set demangle-style gnu-v3

# Pagination settings (useful for long outputs)
set pagination off
