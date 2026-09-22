import { CHAPTERS } from './engine.js';
import { buildMatrix, identify, groupOf, groupRows, focusLanes, faceCells, registerCells, loadFootprint, destCoordinates } from './layout.js';
import { referenceContent } from './reference.js';

const $ = id => document.getElementById(id);
const esc = value => String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
let seed = 42, preset = 'coordinates', focusedRow = 0, matrix = buildMatrix(seed, preset), trace = matrix[focusedRow];
let selectedTile = 0, selectedFace = 0, selectedRegister = 0, lastFaceKey = null;
let index = 0, playing = false, timer = null, mode = 'guided', tracked = null, previousIndex = 0;
let sources = {}, animationHandles = [];
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const board = document.querySelector('.board');
const types = { intro: 'START HERE', unpack: 'UNPACK + TRANSPOSE', config: 'CONFIGURE', phase: 'NEW STAGE', checkpoint: 'MILESTONE', load: 'DEST → REGISTERS', store: 'REGISTERS → DEST', swap: 'COMPARE + EXCHANGE', transpose: 'CROSS-LANE MOVEMENT', result: 'TOP-K COMPLETE' };
const ops = {
  intro: ['↓', 'Ready to load', 'Start with values in L1.'], unpack: ['↳', 'UNPACK / A2D', 'Transpose and stage in DEST.'],
  config: ['⚙', 'SFPCONFIG', 'Set the SFPU control bits.'], phase: ['⌁', 'Bitonic network', 'Opposite directions build a bitonic sequence.'],
  checkpoint: ['✓', 'Stage complete', 'The intermediate result is in DEST.'], load: ['↓', 'SFPLOAD', 'DEST → local registers'],
  store: ['↑', 'SFPSTORE', 'Local registers → DEST'], swap: ['⇅', 'SFPSWAP', 'Compare within each lane.'],
  transpose: ['⤨', 'SFPTRANSP', 'Move across registers and lanes.'], result: ['✓', 'Top-32 ready', 'Return the first value and index tiles.'],
};
function guidedIndices() {
  return trace.events.flatMap((e, i) => {
    if (e.chapter === 'stage' || ['phase', 'checkpoint', 'result'].includes(e.kind)) return [i];
    // Expand the first working set, including its load, transpose, comparisons,
    // replay and store. Other working sets are computed and summarized.
    const first = e.loaded[0];
    if (first === 0 || (e.chapter === 'merge' && first === 4)) return [i];
    return [];
  });
}
let guided = guidedIndices();
const route = () => mode === 'guided' ? guided : trace.events.map((_, i) => i);
function routePosition() {
  const list = route();
  const p = list.findIndex(i => i >= index);
  return p < 0 ? list.length - 1 : p;
}
function chip(item, className, extra = '') {
  if (!item) return '';
  item = item.row === undefined ? identify(item, focusedRow) : item;
  return `<button class="datum pool-${item.pool} ${className}${item.row === focusedRow ? ' row-focus' : ''}${tracked === item.id ? ' tracked' : ''}" data-index="${item.id}" data-source-row="${item.row}" ${extra} title="Value ${item.value} · original input[${item.row}, ${item.index}]" aria-label="Value ${item.value}, original row ${item.row}, column ${item.index}">${item.lane !== undefined ? `<small class="lane-id">L${item.lane}</small>` : ''}<span class="datum-value">${item.value}</span></button>`;
}
function capturePositions() {
  const positions = { reg: new Map(), dst: new Map() };
  board.querySelectorAll('.reg-datum').forEach(el => positions.reg.set(Number(el.dataset.index), el.getBoundingClientRect()));
  board.querySelectorAll('.memory-datum').forEach(el => positions.dst.set(Number(el.dataset.index), el.getBoundingClientRect()));
  return positions;
}
function animateMovement(before, e, animate) {
  animationHandles.forEach(a => a.cancel()); animationHandles = [];
  const overlay = $('movement-overlay'); overlay.innerHTML = '';
  if (!animate || reducedMotion) return;
  const duration = Math.max(180, 780 / Number($('speed').value));
  const boardRect = board.getBoundingClientRect();
  const paths = [];
  const targetSelector = e.kind === 'store' ? '.memory-datum' : '.reg-datum';
  if (!['store', 'load', 'swap', 'transpose', 'unpack'].includes(e.kind)) return;
  const selector = e.kind === 'unpack' ? '.memory-datum' : targetSelector;
  board.querySelectorAll(selector).forEach(el => {
    const id = Number(el.dataset.index), to = el.getBoundingClientRect();
    const from = (e.kind === 'load' || e.kind === 'unpack' ? before.dst : before.reg).get(id);
    if (!from) return;
    if (e.kind === 'store' && (!groupRows(focusedRow).includes(Number(el.dataset.sourceRow)) || !e.loaded.includes(Number(el.dataset.position)))) return;
    const dx = from.left - to.left, dy = from.top - to.top;
    if (Math.abs(dx) + Math.abs(dy) < 2) return;
    const x1 = from.left + from.width / 2 - boardRect.left, y1 = from.top + from.height / 2 - boardRect.top;
    const x2 = to.left + to.width / 2 - boardRect.left, y2 = to.top + to.height / 2 - boardRect.top;
    if (Number(el.dataset.sourceRow) === focusedRow) paths.push(`<path class="flow-path ${e.kind === 'transpose' ? 'transpose-path' : e.kind === 'load' || e.kind === 'store' ? 'load-path' : ''}" d="M ${x1} ${y1} C ${x1 + 15} ${(y1 + y2) / 2}, ${x2 + 15} ${(y1 + y2) / 2}, ${x2} ${y2}"/>`);
    animationHandles.push(el.animate([
      { transform: `translate(${dx}px, ${dy}px) scale(${from.width / to.width}, ${from.height / to.height})`, zIndex: 15, opacity: .8 },
      { transform: 'translate(0, 0) scale(1)', zIndex: 15, opacity: 1 },
    ], { duration, easing: 'cubic-bezier(.22,.7,.2,1)' }));
  });
  overlay.innerHTML = paths.join('');
  if (paths.length) animationHandles.push(overlay.animate([{ opacity: 0 }, { opacity: 1, offset: .22 }, { opacity: .12 }], { duration: duration * 1.4, fill: 'forwards' }));
}
function renderDest(e) {
  const intro = e.kind === 'intro', footprint = loadFootprint(e, focusedRow, selectedRegister);
  const facePositions = ['top left', 'top right', 'bottom left', 'bottom right'];
  $('memory-title').textContent = intro ? 'Input tile in L1' : 'Destination register file';
  $('memory-subtitle').textContent = intro ? 'face layout before unpack' : 'after unpack transpose';
  $('memory-description').textContent = intro ? `Green cells are original matrix row ${focusedRow}. Step once to see this row become a DEST column.` : `Original row ${focusedRow} is now a DEST column. Each cell below is one value; every column is shown.`;
  $('dest-overview').innerHTML = [0, 1].map(tile => `<div class="dest-tile"><div class="dest-tile-label">${intro ? 'INPUT' : 'DEST'} TILE ${tile}<small>${intro ? `matrix columns ${tile * 32}–${tile * 32 + 31}` : `rows ${tile * 64}–${tile * 64 + 63}`}</small></div><div class="face-buttons">${[0, 1, 2, 3].map(face => {
    const active = selectedTile === tile && selectedFace === face;
    const touched = [0, 1, 2, 3].some(r => { const f = loadFootprint(e, focusedRow, r); return f && f.tile === tile && f.face === face; });
    return `<button data-face="${tile}:${face}" class="face-button${active ? ' selected' : ''}${touched ? ' used' : ''}" style="grid-row:${Math.floor(face / 2) + 1};grid-column:${face % 2 + 1}" aria-label="${intro ? 'Input' : 'DEST'} tile ${tile}, face ${face}, ${facePositions[face]}" aria-pressed="${active}"><strong>F${face}</strong><span>${intro ? '16 × 16 values' : `D${tile * 64 + face * 16}–D${tile * 64 + face * 16 + 15}`}</span><small>${facePositions[face]}</small></button>`;
  }).join('')}</div></div>`).join('');
  $('dest-face-title').textContent = `${intro ? 'Input' : 'DEST'} tile ${selectedTile} · face ${selectedFace} · 256 values`;
  $('dest-face-origin').textContent = intro ? `${facePositions[selectedFace]} of the input tile · within-face row-major layout` : `${facePositions[selectedFace]} of the DEST tile · contains transposed input face ${[0, 2, 1, 3][selectedFace]}`;
  const cells = faceCells(matrix, index, focusedRow, selectedTile, selectedFace);
  let html = `<span class="dest-axis corner">${intro ? 'r / c' : 'D / c'}</span>` + Array.from({ length: 16 }, (_, c) => `<span class="dest-axis col-axis${footprint?.columns.includes(c) ? ' selected-column' : ''}">${c}</span>`).join('');
  for (let r = 0; r < 16; r++) {
    const rowAddress = selectedTile * 64 + selectedFace * 16 + r;
    html += `<span class="dest-axis row-axis${r % 4 === 0 ? ' group-start' : ''}">${intro ? r : 'D' + rowAddress}</span>`;
    for (let c = 0; c < 16; c++) {
      const item = cells[r * 16 + c];
      const addressed = !intro && footprint && footprint.rows.includes(rowAddress) && footprint.columns.includes(c);
      const lane = addressed ? (rowAddress - footprint.rows[0]) * 8 + Math.floor(c / 2) : '';
      html += chip(item, `memory-datum dest-datum${addressed ? ' addressed' : ''}${r % 4 === 0 ? ' group-start' : ''}`, `data-position="${item.position}" data-dest-row="${rowAddress}" data-dest-col="${c}" data-load-lane="${lane}"`);
    }
  }
  $('dest-face').innerHTML = html;
  const choices = [0, 1, 2, 3].map(r => `<button class="load-choice${r === selectedRegister ? ' selected' : ''}" data-register="${r}" aria-pressed="${r === selectedRegister}">LREG${r}</button>`).join('');
  const mapping = footprint ? `<strong>One SFPLOAD → LREG${selectedRegister}: 32 datums</strong><span>DEST rows <b>${footprint.rows[0]}–${footprint.rows[3]}</b> × ${footprint.parity ? 'odd' : 'even'} columns <b>${footprint.columns.join(', ')}</b></span><small>Effective load address ${footprint.effectiveAddress} · tile ${footprint.tile}, face ${footprint.face}${footprint.tile !== selectedTile || footprint.face !== selectedFace ? ' · select this LREG to reveal its face' : ''}</small>` : '<strong>One SFPLOAD reads 4 rows × 8 columns = 32 datums.</strong><span>Select a register to inspect its load. Start the walkthrough to fill DEST and the LREGs.</span>';
  $('load-mapping').innerHTML = `<div class="load-choice-row"><span>INSPECT A LOAD</span>${choices}</div><div class="load-equation">${mapping}</div>`;
  lastFaceKey = `${selectedTile}:${selectedFace}`;
}
function renderRegisters(e) {
  $('sfpu-description').innerHTML = e.kind === 'result' ? 'Last working set shown below. The sorted output is in <strong>DEST tile 0</strong>.' : `All 32 lanes are shown in each LREG, arranged 4 × 8. Original row <strong>${focusedRow}</strong> occupies lanes <strong>${focusLanes(focusedRow).join(', ')}</strong>. Every number is one datum.`;
  $('registers').innerHTML = Array.from({ length: 4 }, (_, r) => `<section class="lreg-card${selectedRegister === r ? ' selected' : ''}" aria-label="LREG${r}, 32 lanes"><button class="lreg-heading" data-register="${r}"><strong>LREG${r}</strong><span>${e.loaded[r * 4] === undefined ? 'not active' : '32 values'} · lanes 0–31</span><small>inspect load ↗</small></button><div class="lreg-grid">${registerCells(matrix, index, focusedRow, r).map((item, lane) => {
    const sourceRow = groupRows(focusedRow)[lane % 8], rowEvent = matrix[sourceRow].events[index];
    const slot = r * 4 + Math.floor(lane / 8), pair = rowEvent.pairs.find(p => p.a === slot || p.b === slot);
    return `<div class="register-slot full-slot${!item ? ' empty' : ''}${pair ? ' comparing' : ''}${pair?.didSwap ? ' swapping' : ''}" data-slot="${r * 32 + lane}" data-register="${r}" data-lane="${lane}">${item ? chip(item, 'reg-datum') : `<span class="empty-lane">L${lane}</span>`}</div>`;
  }).join('')}</div></section>`).join('');
  const [icon, name, description] = ops[e.kind];
  $('operation-icon').textContent = icon; $('operation-name').textContent = name;
  $('operation-description').textContent = description;
  $('operation-detail').textContent = e.batch || (e.replay ? 'Replay: second pass of 5 instructions' : e.kind === 'swap' ? (e.reverse ? 'Reversed polarity · 0x104' : 'Normal polarity · 0x004') : 'Values L0–3 · indices L4–7 (not drawn)');
  $('transfer-label').innerHTML = e.kind === 'load' ? `SFPLOAD ↓ <span>${e.loaded.length / 4} value loads × 32 = ${e.loaded.length * 8} values</span>` : e.kind === 'store' ? `SFPSTORE ↑ <span>${e.loaded.length * 8} values back to DEST</span>` : 'DEST ↔ SFPU <span>32 datums per vector load or store</span>';
  $('comparison-panel').innerHTML = e.pairs.length ? `<div class="compare-focus-label">Comparisons for original row ${focusedRow}; the other lanes operate in parallel.</div><div class="pair-list">${e.pairs.map(p => `<div class="pair ${p.didSwap ? 'changed' : ''}"><small>LANE ${p.lane + Math.floor((focusedRow % 16) / 2)}</small>${p.x.value} <span style="opacity:.4">↔</span> ${p.y.value}<span class="pair-outcome">${p.didSwap ? 'exchanged' : 'already in place'}</span></div>`).join('')}</div>` :
    `<div class="compare-empty"><span>${e.kind === 'transpose' ? '⤨' : e.kind === 'load' ? '↓' : '⌁'}</span><div>${e.kind === 'transpose' ? 'SFPTRANSP moves values between registers and lane groups. All eight original rows remain independent.' : e.kind === 'load' ? `${e.loaded.length * 8} real values are now visible. Matching indices occupy LREG4–7 and can be inspected by clicking a value.` : e.kind === 'result' ? 'The first DEST value tile contains the output for every original row.' : 'Select a value to trace its original matrix coordinate and current DEST address.'}</div></div>`;
}
function sourceSnippet(e, limit = 5) {
  if (e.instruction) return e.instruction.replaceAll('TTI_', '');
  if (e.kind === 'intro') return `row ${focusedRow}: 64 candidates\nK = 32\nSTABLE_SORT = false\nFUSED = false\nRANK_STAMPED = false`;
  if (e.kind === 'unpack') return '_llk_unpack_A_init_(\n  /* transpose_of_faces */ 1,\n  /* within_face_transpose */ 1,\n  ...\n);';
  if (e.kind === 'load' || e.kind === 'store') {
    const op = e.kind === 'load' ? 'SFPLOAD' : 'SFPSTORE';
    const offsets = e.loaded.filter((_, i) => i % 4 === 0).map((_, r) => loadFootprint(e, focusedRow, r).effectiveAddress);
    return offsets.map((v, r) => `${op}(LREG${r}, …, ${v});`).join('\n') + `\n// indices: same offsets +128\n// shown with current base included`;
  }
  const data = sources[e.source];
  if (data) {
    const lines = data.content.split('\n').slice(e.lines[0] - 1, e.lines[1]);
    const nonEmpty = lines.filter(l => l.trim() && !l.trim().startsWith('//') && !['{', '}'].includes(l.trim()));
    return nonEmpty.slice(0, limit).map(l => l.trim()).join('\n');
  }
  return `${e.fn}(…);\n// Source: lines ${e.lines.join('–')}`;
}
function renderInspector() {
  const panel = $('value-inspector');
  if (tracked === null) {
    panel.innerHTML = '<span class="micro-label">TRACE A NUMBER</span><p>Click any number to follow its value and original index through the network.</p>';
    return;
  }
  const column = tracked % 64, row = Math.floor(tracked / 64), item = matrix[row].original[column], e = matrix[row].events[index];
  const location = e.regs.findIndex(x => x?.index === column), dest = e.dst.findIndex(x => x.index === column);
  const coord = destCoordinates(row, dest);
  panel.innerHTML = `<span class="micro-label">FOLLOWING ONE DATUM: ${item.value}</span><p class="origin">Input[${row}, ${item.index}] = ${item.value}</p><p>${e.kind === 'intro' ? 'Still in the original L1 input.' : `DEST tile <strong>${coord.tile}</strong>, face <strong>${coord.face}</strong><br>DEST row <strong>${coord.address}</strong>, column <strong>${coord.column}</strong>`}${location >= 0 ? `<br>Register: <strong>LREG${Math.floor(location / 4)}</strong> · lane <strong>${focusLanes(row)[location % 4]}</strong><br>Paired index: #${item.index} in LREG${Math.floor(location / 4) + 4}.` : '<br>Not active in the shown registers.'}</p><button id="clear-track" class="text-button">Clear selection ×</button>`;
  $('clear-track').onclick = () => { tracked = null; render(false); };
}
function renderNav(e) {
  const current = CHAPTERS.findIndex(c => c.id === e.chapter);
  $('side-chapters').innerHTML = trace.chapters.map((c, i) => `<button class="side-chapter${i === current ? ' active' : ''}" data-chapter="${i}" ${i === current ? 'aria-current="step"' : ''}><span class="number">0${i + 1}</span>${c.name}<span class="side-check">${i < current ? '✓' : i === current ? '•' : ''}</span></button>`).join('');
  $('chapter-rail').innerHTML = trace.chapters.map((c, i) => `<button class="chapter-button${i === current ? ' active' : ''}" data-chapter="${i}" ${i === current ? 'aria-current="step"' : ''}><span class="step-number">${i < current ? '✓' : '0' + (i + 1)}</span><span><strong>${c.short}</strong><small>${c.detail}</small></span></button>`).join('');
  $('phase-label').textContent = e.phase !== null ? `PHASE ${e.phase} · ${2 ** (e.phase + 1)}-VALUE RUNS` : e.chapter === 'merge' ? 'm_iter = 0 · distance = 32' : e.chapter === 'rebuild' ? 'distance 16 → 8 → 4 → 2 → 1' : e.chapter === 'result' ? '32 VALUES + 32 INDICES' : 'L1 → DEST → SFPU';
}
function render(animate = true) {
  const e = trace.events[index];
  const rowMin = Math.min(...trace.original.map(item => item.value));
  const rowMax = Math.max(...trace.original.map(item => item.value));
  $('row-scope').textContent = `Original row ${focusedRow}: 64 candidates, min ${rowMin}, max ${rowMax}. Each of the 32 original rows gets its own Top-32.`;
  if ($('follow-load').checked) {
    const footprint = loadFootprint(e, focusedRow, selectedRegister);
    if (e.kind === 'result') { selectedTile = 0; selectedFace = Math.floor(focusedRow / 16); }
    else if (footprint) { selectedTile = footprint.tile; selectedFace = footprint.face; }
  }
  if (lastFaceKey !== null && lastFaceKey !== `${selectedTile}:${selectedFace}` && e.kind !== 'unpack') renderDest(e);
  const before = capturePositions();
  renderDest(e); renderRegisters(e); renderNav(e); renderInspector();
  $('event-type').textContent = types[e.kind];
  $('event-number').textContent = String(routePosition() + 1).padStart(2, '0');
  $('event-title').textContent = e.kind === 'result' ? `Top-32 for original row ${focusedRow}.` : e.kind === 'load' ? `Load ${e.loaded.length * 8} values into ${e.loaded.length / 4} LREGs.` : e.title;
  $('event-text').textContent = e.kind === 'load' ? `Each SFPLOAD fills one register with 32 values, drawn from four DEST rows and eight ${focusedRow % 2 ? 'odd' : 'even'} columns. This helper loads ${e.loaded.length / 4} value registers and their matching index registers.` :
    e.text.replaceAll('row 0', `row ${focusedRow}`).replaceAll('four visible pairs', `four pairs for row ${focusedRow}`);
  $('event-why').textContent = e.kind === 'load' ? `${groupRows(focusedRow).join(', ')} are the original matrix rows handled in parallel. Row ${focusedRow} contributes ${e.loaded.length} values across the loaded LREGs. Click an LREG heading to highlight its exact 32 source cells.` :
    e.why.replaceAll('lanes 0, 8, 16 and 24', `lanes ${focusLanes(focusedRow).join(', ')}`);
  $('function-name').textContent = e.fn;
  $('code-preview').textContent = sourceSnippet(e);
  $('source-location').textContent = `${e.source === 'test' ? 'tests/sources/topk_test.cpp' : 'ckernel_sfpu_topk.h'} : ${e.lines.join('–')}`;
  $('skip-note').hidden = !(index - previousIndex > 1 && mode === 'guided');
  $('skip-note').textContent = `${index - previousIndex - 1} repeated operations completed at other DEST addresses. “Every step” exposes the sequence for this row group.`;
  const list = route(), position = routePosition();
  $('timeline').max = list.length - 1; $('timeline').value = position;
  $('timeline').setAttribute('aria-valuetext', `${position + 1} of ${list.length}: ${e.title}`);
  $('progress-label').textContent = `STEP ${position + 1}`; $('progress-total').textContent = `/ ${list.length}`;
  $('previous').disabled = index === 0; $('next').disabled = index === trace.events.length - 1;
  $('checkpoint').disabled = index === trace.events.length - 1;
  $('result-section').hidden = e.chapter !== 'result';
  if (e.chapter === 'result') {
    $('result-row').value = focusedRow;
    $('result-title').textContent = `Top-32 of original row ${focusedRow}`;
    $('result-description').textContent = `This row's 64 input values range from ${rowMin} to ${rowMax}. Its 32 winners run from ${trace.output[0].value} down to ${trace.output.at(-1).value}. TopK acts across columns independently for each row: the full 32 × 64 input produces a 32 × 32 output.`;
    $('result-values').innerHTML = trace.output.map(item => chip(item, 'result-datum')).join('');
    $('output-matrix').innerHTML = `<table class="original-matrix-table"><thead><tr><th scope="col">row / rank</th>${Array.from({ length: 32 }, (_, c) => `<th scope="col">${c + 1}</th>`).join('')}</tr></thead><tbody>${matrix.map((t, row) => `<tr data-output-row="${row}"><th scope="row">${row}</th>${t.output.map(item => `<td>${chip(identify(item, row), 'output-matrix-datum')}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
    // Independent validation only; buildTrace never uses this reference sort.
    const valid = matrix.every(t => {
      const golden = [...t.original].sort((a, b) => b.value - a.value).slice(0, 32);
      return t.output.every((item, i) => item.value === golden[i].value && item.index === golden[i].index);
    });
    $('verification-icon').textContent = valid ? '✓' : '!';
    $('verification-text').textContent = valid ? 'All 32 row results match independent descending-sort references' : 'Validation mismatch — inspect the exported trace.';
  }
  $('mode-note').textContent = mode === 'guided' ? `Following row group ${groupOf(focusedRow)}. Guided mode summarizes repeated loads.` : `Every step: ${trace.events.length} operations for row group ${groupOf(focusedRow)}. Vector loads / stores are grouped.`;
  updatePlaybackButton();
  animateMovement(before, e, animate);
  // A small inspection surface also supports the browser smoke test.
  window.topkLab = { index, mode, event: e, trace, playing, tracked, focusedRow, selectedTile, selectedFace, selectedRegister,
    registers: [0, 1, 2, 3].map(r => registerCells(matrix, index, focusedRow, r)),
    footprint: loadFootprint(e, focusedRow, selectedRegister), goTo: i => goTo(i) };
}
function updatePlaybackButton() {
  $('play-icon').textContent = playing ? 'Ⅱ' : index === trace.events.length - 1 ? '↶' : '▶';
  $('play-label').textContent = playing ? 'Pause walkthrough' : index === trace.events.length - 1 ? 'Replay walkthrough' : 'Play walkthrough';
  $('play').setAttribute('aria-label', playing ? 'Pause animation' : 'Play animation');
}
function stop() { playing = false; clearTimeout(timer); timer = null; updatePlaybackButton(); }
function schedule() {
  clearTimeout(timer);
  if (!playing) return;
  const e = trace.events[index];
  const hold = ['phase', 'checkpoint', 'result', 'intro'].includes(e.kind) ? 3800 : 2300;
  timer = setTimeout(() => {
    const next = route().find(i => i > index);
    if (next === undefined) { stop(); render(false); return; }
    goTo(next, false); schedule();
  }, hold / Number($('speed').value));
}
function goTo(i, pause = true) {
  if (pause) stop();
  previousIndex = index; index = Math.max(0, Math.min(trace.events.length - 1, Number(i)));
  if (index === 0 && previousIndex !== 0) { selectedTile = 0; selectedFace = Math.floor(focusedRow / 16) * 2; }
  else if (previousIndex === 0 && index > 0) selectedFace = [0, 2, 1, 3][selectedFace];
  render(Math.abs(index - previousIndex) <= 1 || mode === 'guided');
}
function step(direction) {
  const list = route();
  const next = direction > 0 ? list.find(i => i > index) : list.findLast(i => i < index);
  if (next !== undefined) goTo(next);
}
function togglePlay() {
  if (playing) { stop(); render(false); return; }
  if (index === trace.events.length - 1) goTo(0);
  playing = true; updatePlaybackButton(); schedule(); render(false);
}
function setMode(next) {
  stop(); mode = next;
  $('guided').classList.toggle('selected', mode === 'guided'); $('full').classList.toggle('selected', mode === 'full');
  $('guided').setAttribute('aria-pressed', String(mode === 'guided')); $('full').setAttribute('aria-pressed', String(mode === 'full'));
  if (mode === 'guided' && !guided.includes(index)) index = guided.findLast(i => i < index) ?? 0;
  previousIndex = index; render(false);
}
function restartInput() { stop(); matrix = buildMatrix(seed, preset); trace = matrix[focusedRow]; guided = guidedIndices(); index = previousIndex = 0; tracked = null; selectedTile = selectedRegister = 0; selectedFace = Math.floor(focusedRow / 16) * 2; render(false); }

function selectRow(row) {
  stop(); focusedRow = row; trace = matrix[focusedRow]; $('focus-row').value = row;
  if (tracked !== null && Math.floor(tracked / 64) !== row) tracked = null;
  if (index === 0) selectedFace = Math.floor(row / 16) * 2;
  else if (!trace.events[index].loaded.length) selectedFace = Math.floor(row / 16);
  render(false);
}

function openDialog(kind) {
  stop();
  const e = trace.events[index];
  let content = referenceContent(kind, { matrix, trace, focusedRow, chip });
  if (kind === 'source') {
    const data = sources[e.source];
    const filename = data?.path || (e.source === 'test' ? 'tests/sources/topk_test.cpp' : 'tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_topk.h');
    const start = e.lines[0], end = e.lines[1];
    content = `<h2>${esc(e.fn)}</h2><p><code>${esc(filename)}</code><br>Local source · lines ${start}–${end}</p>`;
    if (data) content += `<pre class="dialog-source">${data.content.split('\n').slice(start - 1, end).map((line, j) => `<span class="source-line"><em>${start + j}</em>${esc(line)}</span>`).join('')}</pre><p>The workbench follows the non-stable branches. Source excerpts are read directly from the local checkout.</p>`;
    else content += '<p class="source-warning">Source could not be loaded. Start the site with python3 server.py so the source endpoint is available.</p>';
  }
  $('dialog-content').innerHTML = content;
  $('info-dialog').showModal();
}

document.addEventListener('click', event => {
  const registerButton = event.target.closest('button[data-register]');
  if (registerButton) { selectedRegister = Number(registerButton.dataset.register); $('follow-load').checked = true; render(false); return; }
  const faceButton = event.target.closest('[data-face]');
  if (faceButton) { [selectedTile, selectedFace] = faceButton.dataset.face.split(':').map(Number); $('follow-load').checked = false; render(false); return; }
  const chapterButton = event.target.closest('[data-chapter]');
  if (chapterButton) { $('info-dialog').close(); goTo(trace.chapters[Number(chapterButton.dataset.chapter)].start); return; }
  const dialogButton = event.target.closest('[data-dialog]');
  if (dialogButton) { openDialog(dialogButton.dataset.dialog); return; }
  const datum = event.target.closest('[data-index]');
  if (datum) {
    const id = Number(datum.dataset.index), row = Math.floor(id / 64);
    const alreadyTracked = tracked === id;
    if (row !== focusedRow) selectRow(row);
    tracked = alreadyTracked ? null : id;
    render(false);
  }
});
$('focus-row').innerHTML = Array.from({ length: 32 }, (_, r) => `<option value="${r}">${r}</option>`).join('');
$('result-row').innerHTML = $('focus-row').innerHTML;
$('focus-row').onchange = event => selectRow(Number(event.target.value));
$('result-row').onchange = event => selectRow(Number(event.target.value));
$('follow-load').onchange = () => render(false);
$('original-button').onclick = () => openDialog('original');
$('source-button').onclick = () => openDialog('source');
$('close-dialog').onclick = () => $('info-dialog').close();
$('info-dialog').addEventListener('click', e => { if (e.target === $('info-dialog')) { const r = e.target.getBoundingClientRect(); if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) e.target.close(); } });
$('play').onclick = togglePlay;
$('previous').onclick = () => step(-1); $('next').onclick = () => step(1); $('reset').onclick = () => goTo(0);
$('speed').onchange = schedule;
$('timeline').oninput = e => goTo(route()[Number(e.target.value)]);
$('guided').onclick = () => setMode('guided'); $('full').onclick = () => setMode('full');
$('checkpoint').onclick = () => { const next = trace.events.findIndex((e, i) => i > index && e.checkpoint); if (next >= 0) goTo(next); };
$('preset').onchange = e => { preset = e.target.value; restartInput(); };
$('shuffle').onclick = () => { seed += 127; if (preset === 'descending' || preset === 'coordinates') { preset = 'mixed'; $('preset').value = preset; } restartInput(); };
$('export').onclick = () => {
  const blob = new Blob([JSON.stringify({ description: 'Per-row Blackhole non-stable K=32; full matrix results and selected row trace, not a cycle trace', focusedRow, inputMatrix: matrix.map(t => t.original.map(x => x.value)), outputMatrix: matrix.map(t => t.output.map(x => x.value)), outputIndexMatrix: matrix.map(t => t.output.map(x => x.index)), output: trace.output, events: trace.events }, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob), anchor = document.createElement('a'); anchor.href = url; anchor.download = 'blackhole-topk-trace.json'; anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
};
document.addEventListener('keydown', e => {
  if ($('info-dialog').open || ['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON', 'A'].includes(e.target.tagName)) return;
  if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
  if (e.key === 'ArrowRight') { e.preventDefault(); step(1); }
  if (e.key === 'ArrowLeft') { e.preventDefault(); step(-1); }
  if (e.key === 'Home') { e.preventDefault(); goTo(0); }
  if (e.key === 'End') { e.preventDefault(); goTo(trace.events.length - 1); }
});
document.addEventListener('visibilitychange', () => { if (document.hidden) { stop(); render(false); } });
window.addEventListener('resize', () => { $('movement-overlay').innerHTML = ''; });
render(false);
fetch('api/source').then(r => { if (!r.ok) throw new Error('Source endpoint unavailable'); return r.json(); }).then(data => { sources = data; $('code-preview').textContent = sourceSnippet(trace.events[index]); }).catch(() => { /* The simulation works independently of the optional source endpoint. */ });
