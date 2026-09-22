import { identify, focusLanes, groupRows } from './layout.js';

const isaBase = 'https://github.com/tenstorrent/tt-isa-documentation/blob/main/';
const isaLinks = `<a href="${isaBase}BlackholeA0/TensixTile/TensixCoprocessor/SFPLOAD.md" target="_blank" rel="noreferrer">SFPLOAD</a> · <a href="${isaBase}BlackholeA0/TensixTile/TensixCoprocessor/SFPSTORE.md" target="_blank" rel="noreferrer">SFPSTORE</a> · <a href="${isaBase}BlackholeA0/TensixTile/TensixCoprocessor/SFPSWAP.md" target="_blank" rel="noreferrer">SFPSWAP</a> · <a href="${isaBase}WormholeB0/TensixTile/TensixCoprocessor/SFPTRANSP.md" target="_blank" rel="noreferrer">SFPTRANSP (shared with Blackhole)</a>`;

export function referenceContent(kind, { matrix, trace, focusedRow, chip }) {
  if (kind === 'lanes') return `<h2>One LREG. Thirty-two individual values.</h2>
    <p>A value register is drawn as four groups of eight lanes. Every cell is one 32-bit datum. Lane numbers are the small L0…L31 labels; they are not register numbers.</p>
    <table class="reference-table"><tr><th>LANES IN ONE LREG</th><th>DEST POSITIONS READ BY AN EVEN-COLUMN LOAD</th></tr>
    ${[0, 1, 2, 3].map(r => `<tr><td>${r * 8}–${r * 8 + 7}</td><td>DEST row base+${r}, columns 0, 2, 4, 6, 8, 10, 12, 14</td></tr>`).join('')}</table>
    <p>An odd-column load instead selects columns 1, 3, 5, …, 15. Both instructions still load 32 datums. Four value loads fill LREG0–3 with 128 datums.</p>
    <h3>Why one original row occupies four lanes</h3><p>Input unpack transposes the tile, including face placement. An original matrix row becomes a DEST column. Reading four DEST rows gives four candidates from that original row. Original row <strong>${focusedRow}</strong> appears in lanes <strong>${focusLanes(focusedRow).join(', ')}</strong>. The other lanes contain real values from rows ${groupRows(focusedRow).filter(r => r !== focusedRow).join(', ')}.</p>
    <div class="formula">DEST[base + floor(lane / 8), 2 × (lane % 8) + parity]<br>→ LREG[lane]</div>
    <h3>How comparisons reach other positions</h3><p><code>SFPSWAP</code> compares two registers at the same lane. <code>SFPTRANSP</code> exchanges register number and lane-group number: (register r, lane 8g+b) becomes (register g, lane 8r+b). All eight independent original rows participate. Values from different original rows do not mix.</p>
    <p>Values use LREG0–3. Corresponding indices occupy LREG4–7 and follow the same movements; they are available in the value inspector. References: ${isaLinks}.</p>`;

  if (kind === 'scope') return `<h2>The DEST layout, with every lane visible.</h2>
    <p>This walkthrough follows the ordinary, unfused, non-stable <strong>largest-first K=32</strong> path in your local <code>tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_topk.h</code>. The example contains 32 matrix rows and 64 value columns.</p>
    <table class="reference-table"><tr><th>HARDWARE STRUCTURE</th><th>WHAT THE SITE SHOWS</th></tr>
    <tr><td>Two 32 × 32 value tiles</td><td>Two tile maps, each with four face selectors. The selected face shows all 256 datums and all 16 columns.</td></tr>
    <tr><td>DEST faces in row-major face order</td><td>Face 0: rows 0–15; face 1: 16–31; face 2: 32–47; face 3: 48–63. Tile 1 adds 64. D labels show logical ISA DEST row addresses relative to the value region, not SRAM byte addresses.</td></tr>
    <tr><td>Initial unpack transpose</td><td>Input faces F0, F1, F2, F3 become F0ᵀ, F2ᵀ, F1ᵀ, F3ᵀ in the four DEST face slots. The highlighted original row visibly turns into a column.</td></tr>
    <tr><td>LREG0–3, 32 lanes each</td><td>Every value lane is drawn. Each original row contributes four lanes per register. No displayed number represents multiple values.</td></tr>
    <tr><td>Four face/column passes</td><td>The stepper follows the selected row group: even rows 0–14, odd rows 1–15, even rows 16–30, or odd rows 17–31. Earlier groups show their completed state for the entry point; later groups wait at its input state.</td></tr>
    <tr><td>Index tiles and LREG4–7</td><td>Indices are carried by the model and available when a value is selected. The face grids and LREG cards show values.</td></tr>
    <tr><td>Final rebuild for K=32</td><td>The header's default branch sets an inner total_datums_to_compare=64. Both halves are visited even when skip_second=1. Only the first value tile is returned.</td></tr></table>
    <h3>What the animation groups</h3><p>A load or store event groups the helper's vector instructions. Use the LREG selector to inspect one individual instruction's 32-cell footprint. Other row-group passes, replay recording, address-counter instructions, and synchronization are summarized; animation time is not hardware cycle time.</p>
    <p>The numbered input is FP32-exact <code>value = row × 64 + column</code>. Shuffled, signed, and descending presets use distinct values per row. A register marked “not active” has not been loaded for this working set; stale hardware contents are omitted. Stable, fused, rank-stamped, UInt16-value and special-value modes are outside this lesson. No accelerator execution is claimed.</p>
    <h3>Sources</h3><p>Local source is read directly from the header and <code>tests/sources/topk_test.cpp</code>. The register geometry follows the primary ISA references: ${isaLinks}.</p>`;

  if (kind === 'map') return `<h2>The path through the code.</h2><p>The caller stages a tile pair. The SFPU builds a bitonic sequence for each row, selects its winners, and orders them.</p>
    ${trace.chapters.map((c, i) => `<button class="map-card" data-chapter="${i}"><span><strong>0${i + 1} · ${c.name}</strong><small>${['_llk_unpack_A_ → A2D → _init_topk', '_bitonic_topk_phases_steps(0, 4, 0, 0, 0)', '_bitonic_topk_merge(m_iter=0, k=32)', '_bitonic_topk_rebuild(0, 0, 32, 5, 1)', 'pack value tile 0 + matching index tile'][i]}</small></span><span>→</span></button>`).join('')}
    <h3>Inside the driver</h3><p><code>bitonic_topk_load16</code> → phase helper with <code>SFPSWAP</code> and <code>SFPTRANSP</code> → <code>bitonic_topk_store16</code>. Each face/column pass handles eight original rows simultaneously. The load16 helper brings 16 candidates from each of those rows into four 32-lane value registers.</p>`;

  if (kind === 'original') return `<h2>The complete original 32 × 64 matrix.</h2><p>Each line below is one original matrix row; each column is a real matrix column. Green cells mark row ${focusedRow}. Scroll horizontally to see all 64 columns. Values are stored in tilized face order in L1.</p>
    <div class="original-matrix-scroll"><table class="original-matrix-table"><thead><tr><th>r / c</th>${Array.from({ length: 64 }, (_, c) => `<th>${c}</th>`).join('')}</tr></thead><tbody>${matrix.map((t, row) => `<tr><th>${row}</th>${t.original.map(item => `<td>${chip(identify(item, row), 'original-datum')}</td>`).join('')}</tr>`).join('')}</tbody></table></div>
    <h3>After the initial transpose</h3><p>An original row becomes a DEST column. For original matrix coordinate (r, c): tile = floor(c / 32), face = 2 × floor((c % 32) / 16) + floor(r / 16), within-face DEST row = c % 16, and DEST column = r % 16. The main diagram uses this mapping directly.</p>`;
  return '';
}
