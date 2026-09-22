import assert from 'node:assert/strict';
import { buildMatrix, destCoordinates, sourceCoordinates, groupRows, focusLanes,
  registerCells, faceCells, loadFootprint, eventForRow, boundaryIndices } from './layout.js';

const matrix = buildMatrix(42, 'coordinates');
assert.deepEqual(groupRows(0), [0, 2, 4, 6, 8, 10, 12, 14]);
assert.deepEqual(groupRows(31), [17, 19, 21, 23, 25, 27, 29, 31]);
assert.deepEqual(focusLanes(2), [1, 9, 17, 25]);
assert.deepEqual(focusLanes(31), [7, 15, 23, 31]);

// Every matrix coordinate maps to one unique DEST cell and back.
const addresses = new Set();
for (let row = 0; row < 32; row++) for (let col = 0; col < 64; col++) {
  const d = destCoordinates(row, col);
  assert.deepEqual(sourceCoordinates(d.tile, d.face, d.row, d.column), { row, position: col });
  assert.equal(d.address, d.tile * 64 + d.face * 16 + d.row);
  addresses.add(`${d.address}:${d.column}`);
}
assert.equal(addresses.size, 2048);

// Input face rows turn into DEST columns, including the exchange of faces 1/2.
for (let tile = 0; tile < 2; tile++) for (let face = 0; face < 4; face++) {
  const before = faceCells(matrix, 0, 0, tile, face);
  const after = faceCells(matrix, 1, 0, tile, [0, 2, 1, 3][face]);
  for (let r = 0; r < 16; r++) for (let c = 0; c < 16; c++) {
    assert.equal(before[r * 16 + c].value, (Math.floor(face / 2) * 16 + r) * 64 + tile * 32 + face % 2 * 16 + c);
    assert.equal(before[r * 16 + c].id, after[c * 16 + r].id);
  }
}

let vectorLoads = 0, transposes = 0, stores = 0;
for (const row of [0, 1, 2, 15, 16, 31]) {
  const trace = matrix[row];
  for (const [index, event] of trace.events.entries()) {
    if (event.kind === 'load' || event.kind === 'store') {
      for (let reg = 0; reg < 4; reg++) {
        const f = loadFootprint(event, row, reg);
        if (!f) { assert.ok(registerCells(matrix, index, row, reg).every(v => v === null)); continue; }
        const cells = faceCells(matrix, index, row, f.tile, f.face);
        const lanes = registerCells(matrix, index, row, reg);
        const base = f.effectiveAddress & ~3;
        const parity = (f.effectiveAddress >> 1) & 1;
        // Independently apply the ISA's lane -> DEST row/column formula.
        for (let lane = 0; lane < 32; lane++) {
          const address = base + Math.floor(lane / 8), column = 2 * (lane % 8) + parity;
          const cell = cells.find(c => c.destRow === address && c.destColumn === column);
          assert.ok(cell, `Missing cell at D${address}, c${column}`);
          assert.equal(lanes[lane].id, cell.id, `${event.kind}, event ${index}, row ${row}, LREG${reg}, lane ${lane}`);
          assert.equal(lanes[lane].value, cell.value);
        }
        if (event.kind === 'load') vectorLoads++; else stores++;
      }
    }
    if (event.kind === 'transpose') {
      const before = [0, 1, 2, 3].map(r => registerCells(matrix, index - 1, row, r));
      const after = [0, 1, 2, 3].map(r => registerCells(matrix, index, row, r));
      for (let r = 0; r < 4; r++) for (let g = 0; g < 4; g++) for (let b = 0; b < 8; b++) {
        assert.equal(before[r][8 * g + b]?.id, after[g][8 * r + b]?.id);
      }
      transposes++;
    }
  }
}

// Other row groups reflect the actual driver's earlier/later entry-point passes.
const bounds = boundaryIndices(matrix[0]);
for (const chapter of ['local', 'merge', 'rebuild']) {
  const index = matrix[0].events.findIndex(e => e.chapter === chapter && e.kind === 'load');
  const previous = { local: 0, merge: bounds.local, rebuild: bounds.merge }[chapter];
  assert.equal(eventForRow(matrix, 0, index, 1), matrix[0].events[bounds[chapter]]);
  assert.equal(eventForRow(matrix, 1, index, 1), matrix[1].events[index]);
  assert.equal(eventForRow(matrix, 17, index, 1), matrix[17].events[previous]);
}

for (const model of [matrix, buildMatrix(17, 'mixed')]) {
  for (const trace of model) {
    const golden = [...trace.original].sort((a, b) => b.value - a.value).slice(0, 32);
    assert.deepEqual(trace.output, golden);
  }
}
console.log(`PASS: all 2048 DEST coordinates; ${vectorLoads} vector loads, ${stores} stores, ${transposes} full-lane transposes; row-group ordering and 64 row outputs.`);
