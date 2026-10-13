# Unary stack (F) against its previous head (P) and F with sigmoid and silu on literal constants (Fhi, the gated path),
# every changed unary op at DRAM 1x1x1024x1024 and height-sharded plus model shapes; P2 repeats P (A/A); relu is a
# cheap reference at the same shapes.
ORDERS=("P F Fhi P2" "P2 Fhi F P" "F P P2 Fhi" "Fhi P2 P F")
CASES=""
