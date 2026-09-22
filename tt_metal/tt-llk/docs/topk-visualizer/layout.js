// Coordinate mapping for the full 32 x 64 example, projected onto one
// face/column driver pass. Each pass operates on eight original matrix rows.
import { buildTrace, makeInput, address } from './engine.js';

export const groupOf = row => Math.floor(row / 16) * 2 + row % 2;
export const groupRows = row => Array.from({ length: 8 }, (_, b) => Math.floor(row / 16) * 16 + row % 2 + b * 2);
export const focusLanes = row => [0, 8, 16, 24].map(lane => lane + Math.floor((row % 16) / 2));
export const identify = (item, row) => item && ({ ...item, row, id: row * 64 + item.index });

export function buildMatrix(seed, preset) {
  return Array.from({ length: 32 }, (_, row) => {
    const values = preset === 'coordinates' ? Array.from({ length: 64 }, (_, c) => row * 64 + c)
      : makeInput(seed + row * 104729, preset);
    return buildTrace(values);
  });
}

export function destCoordinates(row, position) {
  const tile = Math.floor(position / 32);
  const face = Math.floor((position % 32) / 16) * 2 + Math.floor(row / 16);
  return { tile, face, row: position % 16, column: row % 16, address: address(position) + Math.floor(row / 16) * 16 };
}

export function sourceCoordinates(tile, face, r, c, transposed = true) {
  return transposed
    ? { row: (face % 2) * 16 + c, position: tile * 32 + Math.floor(face / 2) * 16 + r }
    : { row: Math.floor(face / 2) * 16 + r, position: tile * 32 + (face % 2) * 16 + c };
}

export function boundaryIndices(trace) {
  return {
    local: trace.events.findLastIndex(e => e.chapter === 'local'),
    merge: trace.events.findLastIndex(e => e.chapter === 'merge'),
    rebuild: trace.events.length - 1,
  };
}

// The actual driver visits face/column passes in order 0, 1, 2, 3 within
// each entry point. Show earlier passes finished and later passes waiting.
export function eventForRow(matrix, row, index, focusedRow) {
  const chapter = matrix[focusedRow].events[index].chapter;
  if (chapter === 'stage') return matrix[row].events[index];
  if (chapter === 'result') return matrix[row].events.at(-1);
  if (groupOf(row) === groupOf(focusedRow)) return matrix[row].events[index];
  const bounds = boundaryIndices(matrix[row]);
  const previous = { local: 0, merge: bounds.local, rebuild: bounds.merge };
  const boundary = groupOf(row) < groupOf(focusedRow) ? bounds[chapter] : previous[chapter];
  return matrix[row].events[boundary];
}

export function faceCells(matrix, index, focusedRow, tile, face) {
  const transposed = matrix[focusedRow].events[index].kind !== 'intro';
  return Array.from({ length: 256 }, (_, cell) => {
    const r = Math.floor(cell / 16), c = cell % 16;
    const src = sourceCoordinates(tile, face, r, c, transposed);
    const e = eventForRow(matrix, src.row, index, focusedRow);
    return { ...identify(e.dst[src.position], src.row), destRow: tile * 64 + face * 16 + r,
      faceRow: r, destColumn: c, position: src.position };
  });
}

export function registerCells(matrix, index, focusedRow, register) {
  const rows = groupRows(focusedRow);
  return Array.from({ length: 32 }, (_, lane) => {
    const row = rows[lane % 8], event = matrix[row].events[index];
    const item = event.regs[register * 4 + Math.floor(lane / 8)];
    return item && { ...identify(item, row), lane, register };
  });
}

export function loadFootprint(event, focusedRow, register) {
  const position = event.loaded[register * 4];
  if (position === undefined) return null;
  const coord = destCoordinates(focusedRow, position);
  const parity = focusedRow % 2;
  return { ...coord, position, parity, effectiveAddress: coord.address + parity * 2,
    rows: Array.from({ length: 4 }, (_, i) => coord.address + i),
    columns: Array.from({ length: 8 }, (_, i) => parity + 2 * i) };
}
