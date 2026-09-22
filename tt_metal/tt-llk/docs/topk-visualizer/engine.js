// A one-row projection of the non-stable Blackhole K=32 network.
// Registers contain the four lanes 0, 8, 16, 24. Each object carries the
// independently stored index with its value, modelling hardware index tracking.
export const K = 32;
export const LANES = [0, 8, 16, 24];
export const CHAPTERS = [
  { id: 'stage', name: 'Stage the matrix', short: 'Stage', detail: 'L1 → DEST' },
  { id: 'local', name: 'Build sorted runs', short: 'Local sort', detail: '2 → 4 → 8 → 16 → 32' },
  { id: 'merge', name: 'Keep the winners', short: 'Merge', detail: '64 candidates → 32' },
  { id: 'rebuild', name: 'Order the winners', short: 'Rebuild', detail: 'Bitonic → sorted' },
  { id: 'result', name: 'Read the result', short: 'Result', detail: 'Values + original indices' },
];
export const address = position => Math.floor(position / 16) * 32 + position % 16;
export function makeInput(seed = 42, preset = 'mixed') {
  let state = seed >>> 0;
  const random = () => { state = (1664525 * state + 1013904223) >>> 0; return state / 4294967296; };
  const values = Array.from({ length: 64 }, (_, i) => preset === 'negative' ? i - 39 : i + 1);
  if (preset === 'descending') return values.reverse();
  for (let i = 63; i > 0; i--) { const j = Math.floor(random() * (i + 1)); [values[i], values[j]] = [values[j], values[i]]; }
  return values;
}

export function buildTrace(values) {
  if (values.length !== 64 || values.some(v => !Number.isFinite(v)) || new Set(values).size !== 64) {
    throw new Error('This walkthrough requires 64 distinct finite numbers.');
  }
  const original = values.map((value, index) => ({ value, index, pool: Math.floor(index / 16) }));
  const dst = original.slice();
  let regs = Array(16).fill(null), loaded = [], chapter = 'stage', phase = null;
  let reverse = false, comparisons = 0, swaps = 0;
  const events = [];
  function emit(data) {
    events.push({ chapter, phase, reverse, comparisons, swaps, dst: dst.slice(), regs: regs.slice(),
      loaded: loaded.slice(), pairs: [], touched: [], source: 'kernel', ...data });
  }
  emit({ kind: 'intro', title: 'A matrix row. Sixty-four candidates.',
    text: 'Find the 32 largest values in row 0. The row spans two 32-column tiles. The destination view shows complete 16 × 16 faces, and every value register shows all 32 lanes.',
    why: 'Each other matrix row follows the same independent network. Start with the row in L1, then follow the highlighted DEST positions into the SFPU.',
    fn: 'run_kernel', source: 'test', lines: [136, 155], checkpoint: true });
  emit({ kind: 'unpack', title: 'Turn the row into a DEST column.',
    text: 'The first unpack transposes the faces and the values within each face. After the unpack / A2D copy, this original row is a column across the DEST tile faces.',
    why: 'That layout lets each SFPLOAD gather four consecutive candidates from this row into lanes 0, 8, 16 and 24. Values and original indices occupy separate tiles.',
    fn: '_llk_unpack_A_init_ → _llk_unpack_A_ → A2D', source: 'test', lines: [191, 216] });
  emit({ kind: 'config', title: 'Make each index follow its value.',
    text: '_init_topk() enables hardware index tracking. Values use LREG0–3; their original column indices use LREG4–7.',
    why: 'A compare-and-swap moves both together. Indices are payloads, not comparison keys. This walkthrough uses distinct values and the non-stable path.',
    fn: '_init_topk', lines: [1639, 1642] });

  function load(positions, fn, lines, text) {
    loaded = positions;
    regs = Array(16).fill(null);
    positions.forEach((p, i) => { regs[i] = dst[p]; });
    const offsets = positions.filter((_, i) => i % 4 === 0).map(p => address(p));
    emit({ kind: 'load', title: `Load ${positions.length} candidates into the SFPU.`, fn, lines,
      text: text || 'Each value-register load brings four visible candidates into four lanes. Matching index-register loads bring their original column numbers.',
      why: `Shown DEST value addresses: ${offsets.join(', ')}. Index addresses are each +128. Other lanes process other matrix rows in parallel.`,
      offsets, touched: positions.slice(), batch: positions.length === 16 ? '8 SFPLOAD instructions' : '4 SFPLOAD instructions' });
  }
  function store(fn = 'bitonic_topk_store16', lines = [569, 610]) {
    loaded.forEach((p, i) => { dst[p] = regs[i]; });
    emit({ kind: 'store', title: 'Write the register values back to DEST.', fn, lines,
      text: 'SFPSTORE writes the register values and their paired indices back to the same addressed DEST positions. These are intermediate results; the packer has not written the output yet.',
      why: 'The driver advances the DEST address and reuses these registers for the next DEST addresses. Replay reissues recorded instructions over new data.',
      touched: loaded.slice(), batch: loaded.length === 16 ? '8 SFPSTORE instructions' : '4 SFPSTORE instructions' });
  }
  function transpose(fn, line) {
    const before = regs.slice();
    for (let r = 0; r < 4; r++) for (let l = 0; l < 4; l++) regs[r * 4 + l] = before[l * 4 + r];
    emit({ kind: 'transpose', title: 'Move partners into the same lane.', fn, lines: [line, line],
      text: 'SFPTRANSP exchanges register number and lane-group number. A value at LREG r, lane 8g+b moves to LREG g, lane 8r+b. The offset b stays the same for each original row.',
      why: 'SFPSWAP compares within a lane. This transpose makes a different set of partners available to it. The index registers undergo the same transpose.',
      instruction: 'TTI_SFPTRANSP(0, 0, 0, 0);', beforeRegs: before });
  }
  function swap(c, d, mode, fn, line, extra = {}) {
    const pairs = [];
    for (let lane = 0; lane < 4; lane++) {
      const a = c * 4 + lane, b = d * 4 + lane;
      const x = regs[a], y = regs[b];
      if (!x || !y) continue;
      let maxToC = mode === 'ROWS_02_MAX' ? lane % 2 === 0 : mode === 'ROWS_01_MAX' ? lane < 2 : true;
      if (reverse) maxToC = !maxToC;
      const didSwap = mode === 'UNCONDITIONALLY' || (maxToC ? x.value < y.value : x.value > y.value);
      if (didSwap) { [regs[a], regs[b]] = [regs[b], regs[a]]; swaps++; }
      if (mode !== 'UNCONDITIONALLY') comparisons++;
      pairs.push({ a, b, x, y, didSwap, maxToC, lane: LANES[lane] });
    }
    emit({ kind: 'swap', title: mode === 'UNCONDITIONALLY' ? 'Reverse this register pair.' : `Compare LREG${c} with LREG${d}.`,
      fn, lines: [line, line], pairs,
      text: mode === 'UNCONDITIONALLY' ? 'This unconditional exchange reverses the preceding min/max result for this pair. It creates the opposite direction needed by the bitonic network.' :
        `One SFPSWAP compares the two registers lane by lane. ${pairs.filter(p => p.didSwap).length} of the four visible pairs exchange places; each original index follows its value.`,
      why: mode === 'ALL_ROWS_MAX' ? `With ${reverse ? 'reversal enabled' : 'normal polarity'}, LREG${c} receives the ${reverse ? 'smaller' : 'larger'} value and LREG${d} receives the ${reverse ? 'larger' : 'smaller'} value in each lane.` :
        mode === 'ROWS_02_MAX' ? 'Lane groups 0 and 2 put the larger value in the first register. Groups 1 and 3 do the opposite, building alternating four-value runs.' :
        mode === 'ROWS_01_MAX' ? 'Lane groups 0 and 1 put the larger value in the first register. Groups 2 and 3 do the opposite, building alternating eight-value runs.' : 'This operation always swaps, regardless of the values.',
      instruction: `TTI_SFPSWAP(0, LREG${c}, LREG${d}, ${mode});`, ...extra });
  }
  function ph3(ascending) {
    const fn = 'bitonic_topk_ph3_st4_to_1';
    if (ascending) {
      reverse = true;
      emit({ kind: 'config', title: 'Reverse min/max for an ascending run.', fn, lines: [654, 661],
        text: 'SFPCONFIG writes 0x104: index tracking remains enabled and min/max polarity is reversed. The same instruction body can now produce an ascending run.',
        why: 'Adjacent runs must face opposite directions before the bitonic merge.' });
    }
    for (let pass = 0; pass < 2; pass++) {
      const extra = { replay: pass === 1 };
      swap(0, 2, 'ALL_ROWS_MAX', fn, 694, extra); swap(1, 3, 'ALL_ROWS_MAX', fn, 695, extra);
      swap(0, 1, 'ALL_ROWS_MAX', fn, 698, extra); swap(2, 3, 'ALL_ROWS_MAX', fn, 699, extra);
      transpose(fn, 701);
    }
    if (ascending) {
      reverse = false;
      emit({ kind: 'config', title: 'Restore normal min/max polarity.', fn, lines: [712, 716],
        text: 'SFPCONFIG restores 0x004. Index tracking stays enabled; subsequent comparisons use normal max/min behavior.',
        why: 'The reversal is local to this ascending sequence.' });
    }
  }
  const span = (start, count) => Array.from({ length: count }, (_, i) => start + i);
  function earlyPhase(p) {
    const fn = `bitonic_topk_ph${p}_st${p + 1}_to_1`;
    for (let base = 0; base < 64; base += 16) {
      load(span(base, 16), 'bitonic_topk_load16', [528, 565]);
      if (p === 0) {
        transpose(fn, 810); swap(0, 1, 'ALL_ROWS_MAX', fn, 813); swap(3, 2, 'ALL_ROWS_MAX', fn, 814); transpose(fn, 816);
      } else if (p === 1) {
        transpose(fn, 781); swap(0, 2, 'ROWS_02_MAX', fn, 784); swap(1, 3, 'ROWS_02_MAX', fn, 785);
        swap(0, 1, 'ROWS_02_MAX', fn, 788); swap(2, 3, 'ROWS_02_MAX', fn, 789); transpose(fn, 791);
      } else if (p === 2) {
        swap(0, 1, 'ALL_ROWS_MAX', fn, 744); swap(2, 3, 'ALL_ROWS_MAX', fn, 745); swap(2, 3, 'UNCONDITIONALLY', fn, 746);
        transpose(fn, 748); swap(0, 2, 'ROWS_01_MAX', fn, 751); swap(1, 3, 'ROWS_01_MAX', fn, 752);
        swap(0, 1, 'ROWS_01_MAX', fn, 755); swap(2, 3, 'ROWS_01_MAX', fn, 756); transpose(fn, 758);
      } else ph3(base % 32 === 16);
      store();
    }
  }
  function merge32Network() {
    // Step 5: two 8-value slices from each 32-value run, separated by 16.
    for (let base = 0; base < 64; base += 32) {
      for (let offset = 0; offset < 16; offset += 8) {
        load([...span(base + offset, 8), ...span(base + offset + 16, 8)], 'bitonic_topk_load16', [528, 565],
          'For step 5, load two groups eight values wide, separated by 16 positions. The load16 arguments are (4, 32): the second distance includes the tile-face address gap.');
        const c = base === 0 ? 0 : 2, d = base === 0 ? 2 : 0;
        swap(c, d, 'ALL_ROWS_MAX', 'bitonic_topk_step_N', base === 0 ? 842 : 848);
        swap(c + 1, d + 1, 'ALL_ROWS_MAX', 'bitonic_topk_step_N', base === 0 ? 843 : 849);
        store();
      }
    }
    for (let base = 0; base < 64; base += 16) {
      load(span(base, 16), 'bitonic_topk_load16', [528, 565]); ph3(base >= 32); store();
    }
  }
  chapter = 'local';
  for (let p = 0; p < 5; p++) {
    phase = p;
    emit({ kind: 'phase', title: `Phase ${p}: build runs of ${2 ** (p + 1)}.`, fn: '_bitonic_topk_phases_steps', lines: [972, 986],
      text: `The local-sort driver visits successive DEST addresses, 16 candidates per original row at a time. At the end of this phase, runs of ${2 ** (p + 1)} values alternate descending and ascending.`,
      why: 'A descending run beside an ascending run forms a bitonic sequence. Longer-distance comparisons followed by shorter ones grow the runs.', checkpoint: true });
    if (p < 4) earlyPhase(p); else merge32Network();
    emit({ kind: 'checkpoint', title: p === 4 ? 'Two sorted halves, facing opposite ways.' : `${2 ** (p + 1)}-value runs are ready.`,
      text: p === 4 ? 'DEST positions 0–31 descend; positions 32–63 ascend. Together they form the 64-value bitonic sequence needed for selection.' :
        `Each ${2 ** (p + 1)}-value run is sorted. Its neighboring run points the other way. The next phase doubles the run length.`,
      why: 'Sorting happens through register comparisons and transposes. No CPU sort is used to generate this trace.',
      fn: '_bitonic_topk_phases_steps', lines: [1088, 1096], checkpoint: true, runLength: 2 ** (p + 1) });
  }
  chapter = 'merge'; phase = null;
  emit({ kind: 'phase', title: 'Compare the two halves, 32 positions apart.', fn: '_bitonic_topk_merge', lines: [1234, 1251],
    text: 'For m_iter=0 and K=32, dist=32. Each comparison keeps the larger value in the first half and the smaller in the second.',
    why: 'Because the 64-value sequence is bitonic, these 32 comparisons separate the largest 32 from the smallest 32. The winners still need rebuilding.', checkpoint: true });
  for (let i = 0; i < 32; i += 4) {
    load([...span(i, 4), ...span(i + 32, 4)], 'bitonic_topk_load8', [484, 503],
      'Load four candidates from each half into LREG0 and LREG1. LREG4 and LREG5 hold their original indices. LREG2–3 are unused here; their stale contents are omitted.');
    swap(0, 1, 'ALL_ROWS_MAX', '_bitonic_topk_merge', 1315);
    store('bitonic_topk_store8', [506, 525]);
  }
  emit({ kind: 'checkpoint', title: 'The largest 32 are now in the first half.', fn: '_bitonic_topk_merge', lines: [1313, 1328],
    text: 'Every retained value is at least as large as every value in the second half. The retained half is bitonic, but it is not yet sorted.',
    why: 'Selection and ordering are separate jobs. Rebuild finishes the ordering of the retained values.', checkpoint: true, selected: true });
  chapter = 'rebuild';
  emit({ kind: 'phase', title: 'Rebuild the retained bitonic sequence.', fn: '_bitonic_topk_rebuild', lines: [1559, 1575],
    text: 'The caller requests rebuild(idir=0, m_iter=0, k=32, logk=5, skip_second=1). A distance-16 step followed by the 16-value helper orders the result.',
    why: 'In this header’s K=32 default branch, an inner total_datums_to_compare=64 means both halves are visited even with skip_second=1. Only the first half becomes the output.',
    checkpoint: true });
  merge32Network();
  chapter = 'result';
  emit({ kind: 'result', title: 'Top-32, with every original index intact.', fn: 'run_kernel → _llk_pack_', source: 'test', lines: [551, 563],
    text: 'The first value tile and its index tile contain the result. The pack stage writes these to L1; the test restores row-oriented output by undoing the tile transpose.',
    why: 'The second half contains the lower 32 values and is not part of the Top-K output. Hover or select a value to trace it back to its original input column.',
    checkpoint: true, selected: true });
  return { original, events, output: dst.slice(0, K), chapters: CHAPTERS.map(c => ({ ...c, start: events.findIndex(e => e.chapter === c.id) })) };
}
